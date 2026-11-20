"""What a signed-in user keeps: "My medicines" and saved reports with their share links.

Only the owner can read these; administrators never see them (the activity log only counts
them). A shared report is read without signing in, through its secret link `/shared#<token>`:
the token travels in the URL fragment and then in a POST body, so it never appears in a
server or proxy log.
"""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app import reports
from app.activity import add_event
from app.db import utc_now
from app.deps import CurrentUser, DbSession, PharmaSession, SettingsDep
from app.medicines import PharmaUnavailable, pill_infos
from app.models import Action, Report, SavedMedicine, User
from app.routers.medicines import interactions_report
from app.routers.scan import pharma_unavailable
from app.schemas import (
    ChatExcerpt,
    ReportBody,
    ReportCreate,
    ReportOut,
    ReportPage,
    ReportPill,
    ReportSummary,
    SavedMedicineIn,
    SavedMedicineOut,
    SavedMedicines,
    ShareCreated,
    SharedIn,
    ShareIn,
)
from app.security import hash_token

router = APIRouter(prefix="/api/my", tags=["my data"])
public_router = APIRouter(prefix="/api", tags=["my data"])


# ------------------------------------------------------------------ my medicines


def _medicines(user: User, db: Session, pharma: Session, limit: int) -> SavedMedicines:
    saved = db.scalars(
        select(SavedMedicine)
        .where(SavedMedicine.user_id == user.id)
        .order_by(SavedMedicine.added_at.desc(), SavedMedicine.id.desc())
    ).all()
    try:
        infos = pill_infos(pharma, [s.class_id for s in saved]) if saved else {}
    except PharmaUnavailable as exc:
        raise pharma_unavailable() from exc
    items = []
    for row in saved:
        info = infos.get(row.class_id)
        items.append(
            SavedMedicineOut(
                class_id=row.class_id,
                name=info.name if info else f"class {row.class_id}",
                generic_name=info.generic_name if info else "",
                active_ingredients=info.ingredients if info else [],
                found=info is not None,
                probability=row.probability,
                added_at=row.added_at,
            )
        )
    return SavedMedicines(items=items, limit=limit)


@router.get("/medicines", response_model=SavedMedicines)
async def list_medicines(
    user: CurrentUser, db: DbSession, pharma: PharmaSession, settings: SettingsDep
) -> SavedMedicines:
    return await run_in_threadpool(_medicines, user, db, pharma, settings.max_saved_medicines)


@router.post("/medicines", response_model=SavedMedicines, status_code=status.HTTP_201_CREATED)
async def save_medicine(
    payload: SavedMedicineIn,
    request: Request,
    user: CurrentUser,
    db: DbSession,
    pharma: PharmaSession,
    settings: SettingsDep,
) -> SavedMedicines:
    """Add a pill class to the list (again: its probability is updated)."""

    def run() -> SavedMedicines:
        try:
            known = pill_infos(pharma, [payload.class_id])
        except PharmaUnavailable as exc:
            raise pharma_unavailable() from exc
        if payload.class_id not in known:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown pill class")
        row = db.scalar(
            select(SavedMedicine).where(
                SavedMedicine.user_id == user.id, SavedMedicine.class_id == payload.class_id
            )
        )
        if row is None:
            count = db.scalar(
                select(func.count())
                .select_from(SavedMedicine)
                .where(SavedMedicine.user_id == user.id)
            )
            if (count or 0) >= settings.max_saved_medicines:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    f"Your list is full ({settings.max_saved_medicines} medicines).",
                )
            db.add(
                SavedMedicine(
                    user_id=user.id, class_id=payload.class_id, probability=payload.probability
                )
            )
            add_event(
                db,
                user.id,
                Action.MEDICINE_SAVE,
                request,
                from_scan=payload.probability is not None,
            )
        else:
            row.probability = payload.probability
            row.added_at = utc_now()
        try:
            db.commit()
        except IntegrityError:  # the same class added twice at once: one row is enough
            db.rollback()
        return _medicines(user, db, pharma, settings.max_saved_medicines)

    return await run_in_threadpool(run)


@router.delete("/medicines/{class_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_medicine(class_id: int, user: CurrentUser, db: DbSession) -> None:
    row = db.scalar(
        select(SavedMedicine).where(
            SavedMedicine.user_id == user.id, SavedMedicine.class_id == class_id
        )
    )
    if row is not None:
        db.delete(row)
        db.commit()


@router.delete("/medicines", status_code=status.HTTP_204_NO_CONTENT)
def clear_medicines(user: CurrentUser, db: DbSession) -> None:
    for row in db.scalars(select(SavedMedicine).where(SavedMedicine.user_id == user.id)):
        db.delete(row)
    db.commit()


# ------------------------------------------------------------------ reports


def _own_report(db: Session, user: User, report_id: int) -> Report:
    report = db.get(Report, report_id)
    if report is None or report.user_id != user.id:  # someone else's report: same as missing
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found")
    return report


def _report_out(report: Report) -> ReportOut:
    return ReportOut(**reports.body(report), share=reports.share_state(report, utc_now()))


@router.get("/reports", response_model=ReportPage)
def list_reports(user: CurrentUser, db: DbSession, settings: SettingsDep) -> ReportPage:
    rows = db.scalars(
        select(Report)
        .where(Report.user_id == user.id)
        .order_by(Report.created_at.desc(), Report.id.desc())
    ).all()
    now = utc_now()
    return ReportPage(
        items=[ReportSummary(**reports.summary(r, now)) for r in rows],
        limit=settings.max_reports_per_user,
    )


@router.post("/reports", response_model=ReportOut, status_code=status.HTTP_201_CREATED)
async def create_report(
    payload: ReportCreate,
    request: Request,
    user: CurrentUser,
    db: DbSession,
    pharma: PharmaSession,
    settings: SettingsDep,
) -> ReportOut:
    """Save the interaction check of these pills and ingredients, as it is today.

    The check is run again on the server. Chat answers are kept only with the seal the server
    gave them (`ChatResult.seal`); an answer changed in the browser is refused.
    """
    key: bytes = request.app.state.report_key
    for excerpt in payload.chat:
        if not reports.seal_matches(key, excerpt, excerpt.seal):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "A chat answer does not match the one the assistant gave: ask it again.",
            )

    def run() -> ReportOut:
        count = db.scalar(select(func.count()).select_from(Report).where(Report.user_id == user.id))
        if (count or 0) >= settings.max_reports_per_user:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"You have {settings.max_reports_per_user} reports: delete some first.",
            )
        pill_ids = list(dict.fromkeys(payload.pills))
        try:
            infos = pill_infos(pharma, pill_ids)
            unknown = [c for c in pill_ids if c not in infos]
            if unknown:
                raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown pill class {unknown[0]}")
            ingredients = list(dict.fromkeys(payload.ingredients))
            check = (
                interactions_report(pharma, pill_ids, ingredients)
                if len(pill_ids) + len(ingredients) >= 2
                else None
            )
        except PharmaUnavailable as exc:
            raise pharma_unavailable() from exc
        pills = [
            ReportPill(
                class_id=c,
                name=infos[c].name,
                generic_name=infos[c].generic_name,
                active_ingredients=infos[c].ingredients,
            )
            for c in pill_ids
        ]
        chat = [ChatExcerpt.model_validate(e.model_dump(exclude={"seal"})) for e in payload.chat]
        report = Report(
            user_id=user.id,
            title=payload.title or reports.default_title(pills, ingredients),
            content=reports.content(pills, ingredients, check, chat),
        )
        db.add(report)
        add_event(
            db,
            user.id,
            Action.REPORT_CREATE,
            request,
            items=len(pills) + len(ingredients),
            chat_answers=len(chat),
        )
        db.commit()
        db.refresh(report)
        return _report_out(report)

    return await run_in_threadpool(run)


@router.get("/reports/{report_id}", response_model=ReportOut)
def read_report(report_id: int, user: CurrentUser, db: DbSession) -> ReportOut:
    return _report_out(_own_report(db, user, report_id))


@router.delete("/reports/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_report(report_id: int, request: Request, user: CurrentUser, db: DbSession) -> None:
    report = _own_report(db, user, report_id)
    db.delete(report)
    add_event(db, user.id, Action.REPORT_DELETE, request)
    db.commit()


@router.post("/reports/{report_id}/share", response_model=ShareCreated)
def share_report(
    report_id: int,
    payload: ShareIn,
    request: Request,
    user: CurrentUser,
    db: DbSession,
    settings: SettingsDep,
) -> ShareCreated:
    """A new read-only link, valid `days` days. Any earlier link of this report stops working."""
    report = _own_report(db, user, report_id)
    days = min(payload.days, settings.report_share_max_days)
    token, token_hash = reports.new_share_token()
    report.share_hash = token_hash
    report.share_expires_at = utc_now() + timedelta(days=days)
    report.share_views = 0
    add_event(db, user.id, Action.REPORT_SHARE, request, days=days)
    db.commit()
    state = reports.share_state(report, utc_now())
    assert state is not None
    return ShareCreated(token=token, share=state)


@router.delete("/reports/{report_id}/share", response_model=ReportOut)
def unshare_report(report_id: int, request: Request, user: CurrentUser, db: DbSession) -> ReportOut:
    report = _own_report(db, user, report_id)
    if report.share_hash is not None:
        report.share_hash = None
        report.share_expires_at = utc_now() if report.share_expires_at else None
        add_event(db, user.id, Action.REPORT_UNSHARE, request)
        db.commit()
    return _report_out(report)


@public_router.post(
    "/shared",
    response_model=ReportBody,
    responses={404: {"description": "Unknown, revoked or expired link"}},
    summary="Read a shared report (no sign-in): the token comes from /shared#<token>",
)
def read_shared(payload: SharedIn, db: DbSession) -> ReportBody:
    report = db.scalar(select(Report).where(Report.share_hash == hash_token(payload.token)))
    if report is None or report.share_expires_at is None or report.share_expires_at <= utc_now():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This link is not valid any more.")
    report.share_views += 1
    db.commit()
    return ReportBody(**reports.body(report))
