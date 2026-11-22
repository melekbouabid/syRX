import { useEffect, useRef, useState, type DragEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  api,
  errorMessage,
  type BoxReading,
  type ChatResult,
  type SavedMedicine,
  type DemoImage,
  type InteractionsResult,
  type PillCandidate,
  type ScannedPill,
  type ScanResult,
} from "../services/api";
import { Alert, Badge, Modal, Spinner, staggerIndex, TextField } from "../components";
import { Disclaimer, InteractionReport, SourceList } from "../components/interactions";
import { useDebounced, useQuery } from "../hooks";
import { Icon, type IconName } from "../components/icons";
import { LatencyReadout, ModelCard, ProbBar, ProbabilityBars, ScanCanvas } from "../components/vision";
import { percent, startLiveLoop, type Latency } from "../lib/vision";
import { showDriftBanner } from "../lib/drift";
import { messages, useT } from "../i18n";
import ChatPanel from "./ChatPanel";

const ACCEPTED_TYPES = "image/jpeg,image/png,image/webp";
const MAX_BYTES = 10 * 1024 * 1024;
const LIVE_PAUSE_MS = 400; // between two live frames, after the previous answer arrived
const LIVE_WIDTH = 960; // live frames are scaled down to this width before upload

type Mode = "upload" | "webcam" | "demo";

// Their labels are in src/i18n (scan.modes).
const MODES: { mode: Mode; icon: IconName }[] = [
  { mode: "upload", icon: "upload" },
  { mode: "demo", icon: "image" },
  { mode: "webcam", icon: "camera" },
];

/** A medicine box read by OCR (step 2.7): text read, brand, active ingredients, sources. */
function BoxCard({
  box,
  added,
  onAdd,
}: {
  box: BoxReading;
  added: boolean;
  onAdd: () => void;
}) {
  const t = useT();
  const b = t.scan.box;
  return (
    <article className={`card pill-card${box.recognised ? "" : " pill-card-warn"}`} data-testid="box-card">
      <header className="pill-card-head">
        <span className="pill-number">
          <Icon name="pill" size={16} />
        </span>
        <div>
          <h3>{box.brand ?? (box.recognised ? b.medicineBox : b.notRecognised)}</h3>
          <p className="muted">{b.meta(box.brand_country, percent(box.confidence))}</p>
        </div>
        <Badge tone="warn">{box.recognised ? b.check : b.notRecognisedBadge}</Badge>
      </header>
      <p className="pill-note">
        <Icon name="alert" size={16} /> {b.note}
      </p>
      {box.ingredients.length > 0 ? (
        <ul className="m-0 grid list-none gap-2 p-0" aria-label={b.found}>
          {box.ingredients.map((item) => (
            <li key={item.name} className="rounded-xl border border-line bg-surface-2 p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="cell-title">{item.name}</span>
                <span
                  className="flex w-32 flex-none items-center gap-2 text-[0.88rem] font-semibold tabular-nums"
                  title={b.ocrHint}
                >
                  <ProbBar value={item.confidence} />
                  {percent(item.confidence)}
                </span>
              </div>
              <p className="cell-sub m-0">
                {b.read(item.matched_text, t.scan.method[item.method] ?? item.method, item.in_interaction_table)}
              </p>
              <p className="cell-sub m-0">
                {b.source}{" "}
                <a href={item.url} target="_blank" rel="noreferrer">
                  {item.source}
                </a>{" "}
                · {item.reference}
              </p>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted m-0">{b.nothing}</p>
      )}
      <details className="text-sm">
        <summary className="cursor-pointer font-semibold">{b.textRead(box.text.length)}</summary>
        <p className="muted m-0 mt-2 whitespace-pre-line" data-testid="box-text">
          {box.text.join("\n") || b.noText}
        </p>
      </details>
      <p className="muted m-0 text-xs">{b.rxnav(box.rxnav)}</p>
      {box.recognised && (
        <div className="form-actions">
          <button type="button" className="btn btn-primary" disabled={added} onClick={onAdd}>
            {added ? t.scan.added : t.scan.add}
          </button>
        </div>
      )}
    </article>
  );
}

/** A pill the user has confirmed (or picked by hand), ready for the interaction check. */
interface Confirmed {
  key: string;
  kind: "pill" | "box";
  class_id: number | null; // null for a medicine box: checked by its active ingredients
  name: string;
  generic_name: string;
  active_ingredients: string[];
  probability: number | null; // null when chosen by search rather than from the scan
}

interface Photo {
  url: string; // object URL of exactly what was sent to the scan
  name: string;
  gradcamUrl: string | null; // precomputed Grad-CAM, only for gallery photos (D104)
}

type OnImage = (file: Blob, name: string, gradcamUrl?: string | null) => void;

function latencyOf(result: ScanResult, clientMs: number): Latency {
  return {
    clientMs,
    serverMs: result.timings.server_ms ?? null,
    inferenceMs: result.inference_ms,
  };
}



/** Drag and drop (or click) a photo; checked here before the API checks it again. */
function UploadPanel({ onImage }: { onImage: OnImage }) {
  const t = useT();
  const [dragging, setDragging] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const take = (file: File | undefined) => {
    if (!file) return;
    if (!ACCEPTED_TYPES.split(",").includes(file.type)) {
      setProblem(t.scan.upload.wrongType);
      return;
    }
    if (file.size > MAX_BYTES) {
      setProblem(t.scan.upload.tooLarge);
      return;
    }
    setProblem(null);
    onImage(file, file.name);
  };

  const drop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    take(event.dataTransfer.files[0]);
  };

  return (
    <>
      {problem && <Alert>{problem}</Alert>}
      <label
        data-testid="dropzone"
        data-dragging={dragging}
        className={`grid cursor-pointer justify-items-center gap-2 rounded-card border-2 border-dashed px-5 py-10 text-center transition duration-200 ease-brand ${
          dragging
            ? "scale-[1.01] border-primary bg-primary-soft shadow-lift"
            : "border-line-strong hover:border-primary hover:bg-primary-soft"
        }`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={drop}
      >
        <span className={`text-primary transition-transform duration-200 ${dragging ? "-translate-y-1" : ""}`}>
          <Icon name="upload" size={32} />
        </span>
        <strong className="text-heading">
          {dragging ? t.scan.upload.drop : t.scan.upload.drag}
        </strong>
        <span className="text-sm text-muted">{t.scan.upload.hint}</span>
        <input
          type="file"
          accept={ACCEPTED_TYPES}
          className="sr-only"
          data-testid="scan-file"
          onChange={(event) => {
            take(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
      </label>
    </>
  );
}

function grabFrame(video: HTMLVideoElement, maxWidth: number): Promise<Blob> {
  const scale = Math.min(1, maxWidth / video.videoWidth);
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) =>
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error(messages().scan.live.cameraImage))),
      "image/jpeg",
      0.9,
    ),
  );
}

interface LiveFrame {
  result: ScanResult;
  latency: Latency;
  count: number;
}

/** The detected pills of the last live frame: top-5 bars, confidence, latency. */
function LivePanel({ frame }: { frame: LiveFrame | null }) {
  const t = useT();
  if (!frame) {
    return <p className="m-0 text-sm text-muted">{t.scan.live.waiting}</p>;
  }
  const { result } = frame;
  return (
    <div className="grid content-start gap-3" aria-live="polite">
      <LatencyReadout latency={frame.latency} />
      <p className="m-0 text-sm text-muted">{t.scan.live.frame(frame.count, result.pills.length)}</p>
      {result.pills.length === 0 && (
        <p className="m-0 text-sm text-muted">{result.message ?? t.scan.live.noPill}</p>
      )}
      {result.pills.slice(0, 3).map((pill, index) => (
        <div key={index} className="grid gap-2 rounded-xl border border-line bg-surface p-3">
          <div className="flex items-center gap-2">
            <span className="pill-number">{index + 1}</span>
            <span className="min-w-0 flex-1 truncate font-semibold text-heading">
              {pill.candidates[0]?.name}
            </span>
            {pill.needs_confirmation ? (
              <Badge tone="warn">{t.scan.confirmBadge}</Badge>
            ) : (
              <Badge tone="active">{t.scan.highBadge}</Badge>
            )}
          </div>
          <ProbabilityBars candidates={pill.candidates} />
          <span className="text-xs text-muted">{t.scan.live.detection(percent(pill.detection_confidence))}</span>
        </div>
      ))}
    </div>
  );
}

/** Live camera: continuous inference with boxes and bars, and a capture button. */
function WebcamPanel({ onImage }: { onImage: OnImage }) {
  const t = useT();
  const live_ = t.scan.live;
  const video = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [live, setLive] = useState(true);
  const [frame, setFrame] = useState<LiveFrame | null>(null);

  const start = async () => {
    setProblem(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setProblem(live_.noCamera);
      return;
    }
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 } },
        audio: false,
      });
      setStream(media);
      setLive(true);
    } catch (failure) {
      setProblem(live_.cameraUnavailable(errorMessage(failure)));
    }
  };

  // Attach the stream to the <video>, and stop the camera when leaving the panel.
  useEffect(() => {
    if (video.current && stream) video.current.srcObject = stream;
    return () => stream?.getTracks().forEach((track) => track.stop());
  }, [stream]);

  // Continuous inference: one frame at a time, the next only after the previous answer.
  useEffect(() => {
    if (!stream || !live) {
      setFrame(null);
      return;
    }
    let count = 0;
    return startLiveLoop({
      step: async (signal) => {
        const element = video.current;
        if (!element || element.videoWidth === 0) return null;
        const image = await grabFrame(element, LIVE_WIDTH);
        const { value, clientMs } = await api.scanTimed(image, "live.jpg", signal);
        count += 1;
        return { result: value, latency: latencyOf(value, clientMs), count };
      },
      onResult: setFrame,
      onError: (failure) => {
        setProblem(errorMessage(failure));
        setLive(false);
      },
      pauseMs: LIVE_PAUSE_MS,
    });
  }, [stream, live]);

  const capture = async () => {
    if (!video.current) return;
    try {
      onImage(await grabFrame(video.current, 1920), "webcam.jpg");
    } catch (failure) {
      setProblem(errorMessage(failure));
    }
  };

  const stop = () => {
    setStream(null);
    setFrame(null);
  };

  return (
    <>
      {problem && <Alert>{problem}</Alert>}
      {!stream ? (
        <div className="grid justify-items-center gap-2 rounded-card border-2 border-dashed border-line-strong px-5 py-10 text-center">
          <span className="text-primary">
            <Icon name="camera" size={32} />
          </span>
          <strong className="text-heading">{live_.useCamera}</strong>
          <span className="text-sm text-muted">{live_.cameraHint}</span>
          <button type="button" className="btn btn-primary" onClick={start}>
            {live_.start}
          </button>
        </div>
      ) : (
        <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
          <div className="grid gap-3">
            <div className="relative overflow-hidden rounded-xl bg-surface-2 leading-none">
              <video ref={video} autoPlay playsInline muted className="block h-auto w-full" />
              {frame && live && (
                <ScanCanvas
                  pills={frame.result.pills}
                  width={frame.result.image_width}
                  height={frame.result.image_height}
                  label={live_.predictions}
                />
              )}
              {live && (
                <span className="absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-full bg-black/60 px-2.5 py-1 text-xs font-bold tracking-wide text-white">
                  <span className="size-2 animate-pulse rounded-full bg-red-500" aria-hidden="true" />
                  {live_.badge}
                </span>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <button type="button" className="btn btn-primary" onClick={capture}>
                <Icon name="camera" size={16} />
                {live_.capture}
              </button>
              <label className="check">
                <input type="checkbox" checked={live} onChange={(event) => setLive(event.target.checked)} />
                {live_.predictions}
              </label>
              <button type="button" className="btn btn-ghost btn-sm" onClick={stop}>
                {live_.stop}
              </button>
            </div>
          </div>
          {live ? (
            <LivePanel frame={frame} />
          ) : (
            <p className="m-0 text-sm text-muted">{live_.paused}</p>
          )}
        </div>
      )}
    </>
  );
}

function DemoPanel({ onImage }: { onImage: OnImage }) {
  const t = useT();
  const gallery = useQuery((signal) => api.demoImages(signal), []);
  const [loading, setLoading] = useState<number | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  const pick = async (image: DemoImage) => {
    setLoading(image.id);
    setProblem(null);
    try {
      onImage(await api.demoImage(image), `demo-${image.id}.jpg`, image.gradcam_url);
    } catch (failure) {
      setProblem(errorMessage(failure));
    } finally {
      setLoading(null);
    }
  };

  if (gallery.error) return <Alert>{gallery.error}</Alert>;
  if (!gallery.data) return <Spinner label={t.scan.demo.loading} />;
  if (gallery.data.items.length === 0) {
    return <Alert kind="info">{t.scan.demo.none}</Alert>;
  }
  return (
    <>
      {problem && <Alert>{problem}</Alert>}
      <p className="mb-3 mt-0 text-sm text-muted">
        {gallery.data.note} {t.scan.demo.oneClick}
      </p>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-3">
        {gallery.data.items.map((image, index) => (
          <button
            type="button"
            key={image.id}
            className="group grid animate-[pop-in_0.35s_var(--ease)_backwards] cursor-pointer gap-1.5 rounded-xl border border-line bg-surface p-1.5 text-left text-ink transition duration-200 ease-brand enabled:hover:-translate-y-0.5 enabled:hover:border-primary enabled:hover:shadow-card disabled:cursor-wait disabled:opacity-60"
            style={{ animationDelay: `${index * 40}ms`, ...staggerIndex(index) }}
            onClick={() => pick(image)}
            disabled={loading !== null}
            data-testid={`demo-${image.id}`}
          >
            <span className="relative block">
              <img
                src={image.thumbnail_url}
                alt={t.scan.demo.alt(image.id + 1)}
                loading="lazy"
                className="block aspect-[4/3] w-full rounded-lg object-cover"
              />
              {image.gradcam_url && (
                <span className="absolute right-1.5 top-1.5 rounded-full bg-black/60 px-2 py-0.5 text-[0.7rem] font-semibold text-white">
                  Grad-CAM
                </span>
              )}
            </span>
            <span className="px-1 pb-0.5 text-[0.82rem] leading-snug">
              {loading === image.id
                ? t.scan.demo.loadingOne
                : t.scan.demo.shows(image.expected_name || t.scan.demo.classN(image.class_id))}
              {t.scan.why[image.why] && (
                <span className="block text-xs text-muted">{t.scan.why[image.why]}</span>
              )}
            </span>
          </button>
        ))}
      </div>
    </>
  );
}

/** Search the 150 classes by name, ingredient or imprint, to correct an identification. */
function PillSearch({ onPick }: { onPick: (pill: PillCandidate) => void }) {
  const t = useT();
  const [text, setText] = useState("");
  const q = useDebounced(text.trim());
  const found = useQuery(
    (signal) => (q.length >= 2 ? api.pills(q, signal) : Promise.resolve(null)),
    [q],
  );
  return (
    <div className="pill-search">
      <label className="search-box">
        <Icon name="search" />
        <input
          type="search"
          placeholder={t.scan.search.placeholder}
          aria-label={t.scan.search.label}
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
      </label>
      {found.error && <Alert>{found.error}</Alert>}
      {found.data && (
        <ul className="search-results">
          {found.data.items.map((pill) => (
            <li key={pill.class_id}>
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={() => {
                  onPick({ ...pill, probability: Number.NaN });
                  setText("");
                }}
              >
                <span className="cell-title">{pill.name}</span>
                <span className="cell-sub">
                  {[pill.imprint && t.scan.search.imprint(pill.imprint), pill.color, pill.shape]
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              </button>
            </li>
          ))}
          {found.data.items.length === 0 && <li className="muted">{t.scan.search.none(q)}</li>}
        </ul>
      )}
    </div>
  );
}

/** One detected pill: top-5 to choose from, search for another, confirm. */
function PillChoice({
  pill,
  index,
  confirmed,
  onConfirm,
}: {
  pill: ScannedPill;
  index: number;
  confirmed: boolean;
  onConfirm: (choice: PillCandidate) => void;
}) {
  const t = useT();
  const c = t.scan.choice;
  const [choice, setChoice] = useState<PillCandidate>(pill.candidates[0]);
  const [picked, setPicked] = useState<PillCandidate | null>(null);
  const options = picked ? [picked, ...pill.candidates] : pill.candidates;

  return (
    <article className={`card pill-card${pill.needs_confirmation ? " pill-card-warn" : ""}`}>
      <header className="pill-card-head">
        <span className="pill-number">{index + 1}</span>
        <div>
          <h3>{pill.candidates[0].name}</h3>
          <p className="muted">
            {c.mostLikely(percent(pill.candidates[0].probability), percent(pill.detection_confidence))}
          </p>
        </div>
        {pill.needs_confirmation ? (
          <Badge tone="warn">{t.scan.confirmBadge}</Badge>
        ) : (
          <Badge tone="active">{t.scan.highBadge}</Badge>
        )}
      </header>
      {pill.needs_confirmation && (
        <p className="pill-note">
          <Icon name="alert" size={16} /> {c.notSure}
        </p>
      )}
      <fieldset className="candidates">
        <legend className="sr-only">{c.possible(index + 1)}</legend>
        {options.map((candidate) => {
          const searched = Number.isNaN(candidate.probability);
          return (
            <label
              key={`${candidate.class_id}-${searched}`}
              className={choice.class_id === candidate.class_id ? "candidate candidate-on" : "candidate"}
            >
              <input
                type="radio"
                name={`pill-${index}`}
                checked={choice.class_id === candidate.class_id}
                onChange={() => setChoice(candidate)}
              />
              <span className="candidate-text">
                <span className="cell-title">{candidate.name}</span>
                <span className="cell-sub">
                  {candidate.active_ingredients.length
                    ? candidate.active_ingredients.join(" + ")
                    : c.noIngredient}
                </span>
              </span>
              {searched ? (
                <span className="muted">{c.bySearch}</span>
              ) : (
                <span
                  className="flex w-32 flex-none items-center gap-2 text-[0.88rem] font-semibold tabular-nums"
                  title={c.probability}
                >
                  <ProbBar value={candidate.probability} />
                  {percent(candidate.probability)}
                </span>
              )}
            </label>
          );
        })}
      </fieldset>
      <PillSearch
        onPick={(candidate) => {
          setPicked(candidate);
          setChoice(candidate);
        }}
      />
      <div className="form-actions">
        <button
          type="button"
          className="btn btn-primary"
          disabled={confirmed}
          onClick={() => onConfirm(choice)}
        >
          <Icon name="check" size={16} />
          {confirmed ? t.scan.added : c.confirm}
        </button>
      </div>
    </article>
  );
}





type View = "photo" | "gradcam";

/** The photo with its boxes, or the Grad-CAM of a gallery photo with the same boxes. */
function ResultStage({
  photo,
  result,
  scanning,
  latency,
}: {
  photo: Photo;
  result: ScanResult | null;
  scanning: boolean;
  latency: Latency | null;
}) {
  const t = useT();
  const st = t.scan.stage;
  const [view, setView] = useState<View>("photo");
  const shown = view === "gradcam" && photo.gradcamUrl ? photo.gradcamUrl : photo.url;

  return (
    <div className="card sticky top-[84px] grid gap-3 p-3" data-testid="result-stage">
      {photo.gradcamUrl && (
        <div className="flex gap-1 rounded-xl bg-surface-2 p-1" role="tablist" aria-label={st.pictureShown}>
          {(["photo", "gradcam"] as View[]).map((item) => (
            <button
              key={item}
              type="button"
              role="tab"
              aria-selected={view === item}
              onClick={() => setView(item)}
              className={`flex-1 cursor-pointer rounded-lg border px-3 py-1.5 font-semibold transition ${
                view === item
                  ? "border-line bg-surface text-primary-text shadow-card"
                  : "border-transparent bg-transparent text-muted hover:text-ink"
              }`}
            >
              {item === "photo" ? st.photo : st.gradcam}
            </button>
          ))}
        </div>
      )}
      <div className="relative overflow-hidden rounded-xl bg-surface-2 leading-none">
        <img
          src={shown}
          alt={view === "gradcam" ? st.gradcamAlt(photo.name) : st.photoAlt(photo.name)}
          className="block h-auto w-full"
        />
        {result && (
          <ScanCanvas
            pills={result.pills}
            width={result.image_width}
            height={result.image_height}
            label={st.detected}
          />
        )}
        {scanning && (
          <div className="absolute inset-0 grid place-items-center bg-surface/60" data-testid="scan-busy">
            <Spinner label={st.identifying} />
          </div>
        )}
      </div>
      {view === "gradcam" && photo.gradcamUrl && (
        <p className="m-0 text-sm text-muted">{st.gradcamText}</p>
      )}
      {!photo.gradcamUrl && result && result.kind !== "box" && result.kind !== "unknown" && (
        <p className="m-0 text-sm text-muted" data-testid="gradcam-note">
          {st.gradcamOnly}
        </p>
      )}
      {result && (
        <>
          <p className="m-0 text-sm text-muted" data-testid="scan-kind">
            {result.kind === "box"
              ? st.boxKind(result.timings.ocr_ms != null ? `${Math.round(result.timings.ocr_ms)} ms` : "")
              : result.kind === "unknown"
                ? st.unknownKind
                : st.pillKind(
                    result.pills.length,
                    result.image_width,
                    result.image_height,
                    percent(result.confirm_threshold),
                  )}
          </p>
          {latency && <LatencyReadout latency={latency} />}
        </>
      )}
    </div>
  );
}

/** Data drift alert (GET /monitoring/drift, step 2.8): shown only when the recent photos
 * really differ from the reference photos, re-checked after every scan. */
function DriftBanner({ scans }: { scans: number }) {
  const t = useT();
  const drift = useQuery((signal) => api.drift(signal), [scans]);
  const report = drift.data;
  if (!report || !showDriftBanner(report.status)) return null;
  return (
    <Alert>
      {t.scan.driftBanner(report.drifted.join(", "))}{" "}
      <Link to="/monitoring">{t.scan.seeMonitoring}</Link>.
    </Alert>
  );
}

const MAX_KEPT_ANSWERS = 5;

/** "My medicines": pill classes kept between visits, one click away from the check. */
function SavedMedicinesCard({
  saved,
  error,
  inCheck,
  onAdd,
  onRemove,
}: {
  saved: SavedMedicine[] | null;
  error: string | null;
  inCheck: (classId: number) => boolean;
  onAdd: (medicine: SavedMedicine) => void;
  onRemove: (classId: number) => void;
}) {
  const t = useT();
  const sv = t.scan.saved;
  return (
    <section className="card my-pills" aria-label={sv.title}>
      <header className="pill-card-head">
        <h2>
          <Icon name="bookmark" size={18} /> {sv.title}
        </h2>
        <span className="muted">{sv.kept}</span>
      </header>
      {error && <Alert>{error}</Alert>}
      {saved && saved.length === 0 && <p className="muted">{sv.empty}</p>}
      {saved && saved.length > 0 && (
        <ul className="chip-list">
          {saved.map((medicine) => (
            <li key={medicine.class_id} className="chip">
              <span>
                <strong>{medicine.name}</strong>
                <span className="cell-sub">
                  {medicine.active_ingredients.join(" + ") || sv.noIngredient}
                </span>
              </span>
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                disabled={!medicine.found || inCheck(medicine.class_id)}
                onClick={() => onAdd(medicine)}
              >
                {inCheck(medicine.class_id) ? sv.inCheck : sv.addToCheck}
              </button>
              <button
                type="button"
                className="icon-btn"
                aria-label={sv.remove(medicine.name)}
                onClick={() => onRemove(medicine.class_id)}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function SaveReportDialog({
  pills,
  ingredients,
  answers,
  defaultTitle,
  onClose,
}: {
  pills: number[];
  ingredients: string[];
  answers: ChatResult[];
  defaultTitle: string;
  onClose: () => void;
}) {
  const navigate = useNavigate();
  const t = useT();
  const [title, setTitle] = useState(defaultTitle);
  const [chosen, setChosen] = useState<ChatResult[]>(answers);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const save = async () => {
    setBusy(true);
    setProblem(null);
    try {
      const report = await api.my.createReport({
        title: title.trim(),
        pills,
        ingredients,
        chat: chosen.map((answer) => ({
          question: answer.question,
          answer: answer.answer,
          citations: answer.citations.map(({ n, title: t, url }) => ({ n, title: t, url })),
          seal: answer.seal ?? "",
        })),
      });
      navigate(`/reports/${report.id}`);
    } catch (failure) {
      setProblem(errorMessage(failure));
      setBusy(false);
    }
  };

  return (
    <Modal title={t.scan.dialog.title} onClose={onClose}>
      <p className="muted">{t.scan.dialog.text}</p>
      {problem && <Alert>{problem}</Alert>}
      <TextField label={t.scan.dialog.field} value={title} onChange={setTitle} maxLength={120} />
      {answers.length > 0 && (
        <fieldset className="field">
          <legend>{t.scan.dialog.answers}</legend>
          {answers.map((answer, index) => (
            <label key={index} className="check-row">
              <input
                type="checkbox"
                checked={chosen.includes(answer)}
                onChange={(event) =>
                  setChosen((current) =>
                    event.target.checked ? [...current, answer] : current.filter((a) => a !== answer),
                  )
                }
              />
              {answer.question}
            </label>
          ))}
        </fieldset>
      )}
      <div className="form-actions">
        <button type="button" className="btn btn-ghost" onClick={onClose}>
          {t.common.cancel}
        </button>
        <button type="button" className="btn btn-primary" disabled={busy} onClick={save}>
          {busy && <span className="btn-spinner" aria-hidden="true" />}
          {t.scan.dialog.save}
        </button>
      </div>
    </Modal>
  );
}

export default function ScanPage() {
  const t = useT();
  const [mode, setMode] = useState<Mode>("upload");
  const [photo, setPhoto] = useState<Photo | null>(null);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [latency, setLatency] = useState<Latency | null>(null);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mine, setMine] = useState<Confirmed[]>([]);
  const [report, setReport] = useState<InteractionsResult | null>(null);
  const [checking, setChecking] = useState(false);
  const [scans, setScans] = useState(0);
  const resultRef = useRef<HTMLElement>(null);
  const saved = useQuery((signal) => api.my.medicines(signal), []);
  const [savedList, setSavedList] = useState<SavedMedicine[] | null>(null);
  const [savedError, setSavedError] = useState<string | null>(null);
  const [kept, setKept] = useState<ChatResult[]>([]);
  const [saving, setSaving] = useState(false);
  const mySaved = savedList ?? saved.data?.items ?? null;

  const saveMedicine = async (item: { class_id: number | null; probability: number | null }) => {
    if (item.class_id === null) return;
    setSavedError(null);
    try {
      setSavedList((await api.my.saveMedicine(item.class_id, item.probability)).items);
    } catch (failure) {
      setSavedError(errorMessage(failure));
    }
  };

  const removeSaved = async (classId: number) => {
    setSavedError(null);
    try {
      await api.my.removeMedicine(classId);
      setSavedList((mySaved ?? []).filter((m) => m.class_id !== classId));
    } catch (failure) {
      setSavedError(errorMessage(failure));
    }
  };

  const addSaved = (medicine: SavedMedicine) => {
    setMine((current) => [
      ...current.filter((item) => item.key !== `saved#${medicine.class_id}`),
      {
        key: `saved#${medicine.class_id}`,
        kind: "pill",
        class_id: medicine.class_id,
        name: medicine.name,
        generic_name: medicine.generic_name,
        active_ingredients: medicine.active_ingredients,
        probability: null,
      },
    ]);
    setReport(null);
  };

  const keep = (result: ChatResult) =>
    setKept((current) =>
      current.includes(result)
        ? current.filter((r) => r !== result)
        : [...current, result].slice(-MAX_KEPT_ANSWERS),
    );

  // Release the previous photo's memory when it is replaced or the page is left.
  useEffect(() => () => {
    if (photo) URL.revokeObjectURL(photo.url);
  }, [photo]);

  const identify: OnImage = async (image, name, gradcamUrl = null) => {
    setScanning(true);
    setError(null);
    setResult(null);
    setLatency(null);
    setPhoto({ url: URL.createObjectURL(image), name, gradcamUrl });
    try {
      const { value, clientMs } = await api.scanTimed(image, name);
      setResult(value);
      setLatency(latencyOf(value, clientMs));
      setScans((count) => count + 1);
      window.setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      setScanning(false);
    }
  };

  const confirm = (choice: PillCandidate, pillIndex: number) => {
    const key = `${photo?.url}#${pillIndex}`;
    setMine((current) => [
      ...current.filter((item) => item.key !== key),
      {
        key,
        kind: "pill",
        class_id: choice.class_id,
        name: choice.name,
        generic_name: choice.generic_name,
        active_ingredients: choice.active_ingredients,
        probability: Number.isNaN(choice.probability) ? null : choice.probability,
      },
    ]);
    setReport(null);
  };

  const addBox = (box: BoxReading) => {
    const key = `${photo?.url}#box`;
    const names = box.ingredients.map((item) => item.name);
    setMine((current) => [
      ...current.filter((item) => item.key !== key),
      {
        key,
        kind: "box",
        class_id: null,
        name: box.brand ?? names.join(" + "),
        generic_name: "",
        active_ingredients: names,
        probability: box.confidence,
      },
    ]);
    setReport(null);
  };

  const remove = (key: string) => {
    setMine((current) => current.filter((item) => item.key !== key));
    setReport(null);
  };

  // The same product confirmed twice (front and back of one pill) is checked once.
  const classIds = [
    ...new Set(mine.flatMap((item) => (item.class_id === null ? [] : [item.class_id]))),
  ];
  // Medicine boxes are checked by the active ingredients read on them.
  const boxIngredients = [
    ...new Set(mine.filter((item) => item.kind === "box").flatMap((item) => item.active_ingredients)),
  ];
  const checkable = classIds.length + boxIngredients.length;

  const check = async () => {
    setChecking(true);
    setError(null);
    try {
      setReport(await api.interactions(classIds, boxIngredients));
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      setChecking(false);
    }
  };

  return (
    <>
      <header className="page-header">
        <div className="page-title">
          <span className="page-icon">
            <Icon name="scan" size={24} />
          </span>
          <div>
            <h1>{t.scan.title}</h1>
            <p className="muted">{t.scan.lead}</p>
          </div>
        </div>
      </header>

      <Disclaimer />

      <ModelCard />
      <DriftBanner scans={scans} />

      <section className="card mb-5" aria-label={t.scan.photo}>
        <div className="mb-4 flex flex-wrap gap-1 rounded-xl bg-surface-2 p-1" role="tablist" aria-label={t.scan.photoSource}>
          {MODES.map((item) => (
            <button
              type="button"
              role="tab"
              key={item.mode}
              aria-selected={mode === item.mode}
              className={`inline-flex flex-[1_1_160px] cursor-pointer items-center justify-center gap-2 rounded-lg border px-3 py-2 font-semibold transition duration-200 ${
                mode === item.mode
                  ? "border-line bg-surface text-primary-text shadow-card"
                  : "border-transparent bg-transparent text-muted hover:text-ink"
              }`}
              onClick={() => setMode(item.mode)}
            >
              <Icon name={item.icon} size={16} />
              {t.scan.modes[item.mode]}
            </button>
          ))}
        </div>
        {mode === "upload" && <UploadPanel onImage={identify} />}
        {mode === "webcam" && <WebcamPanel onImage={identify} />}
        {mode === "demo" && <DemoPanel onImage={identify} />}
      </section>

      {error && <Alert>{error}</Alert>}

      {photo && (
        <section
          className="mb-5 grid items-start gap-5 md:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]"
          ref={resultRef}
          aria-label={t.scan.result}
          aria-busy={scanning}
          data-testid="scan-result"
        >
          <ResultStage key={photo.url} photo={photo} result={result} scanning={scanning} latency={latency} />
          <div className="scan-pills">
            {result?.message && <Alert kind="info">{result.message}</Alert>}
            {result?.kind === "box" && result.box && (
              <BoxCard
                box={result.box}
                added={mine.some((item) => item.key === `${photo.url}#box`)}
                onAdd={() => result.box && addBox(result.box)}
              />
            )}
            {result?.pills.map((pill, index) => (
              <PillChoice
                key={`${photo.url}-${index}`}
                pill={pill}
                index={index}
                confirmed={mine.some((item) => item.key === `${photo.url}#${index}`)}
                onConfirm={(choice) => confirm(choice, index)}
              />
            ))}
            {result && <SourceList sources={result.sources} />}
          </div>
        </section>
      )}

      <SavedMedicinesCard
        saved={mySaved}
        error={savedError ?? saved.error}
        inCheck={(classId) => mine.some((item) => item.class_id === classId)}
        onAdd={addSaved}
        onRemove={(classId) => void removeSaved(classId)}
      />

      <section className="card my-pills" aria-label={t.scan.mine.title}>
        <header className="pill-card-head">
          <h2>{t.scan.mine.title}</h2>
          <span className="muted">
            {t.scan.mine.counts(mine.length, classIds.length, boxIngredients.length)}
          </span>
        </header>
        {mine.length === 0 ? (
          <p className="muted">{t.scan.mine.empty}</p>
        ) : (
          <ul className="chip-list">
            {mine.map((item) => (
              <li key={item.key} className="chip">
                <span>
                  <strong>{item.name}</strong>
                  <span className="cell-sub">
                    {item.active_ingredients.join(" + ") || t.scan.saved.noIngredient}
                    {item.probability !== null ? ` · ${percent(item.probability)}` : t.scan.mine.bySearch}
                  </span>
                </span>
                {item.class_id !== null && (
                  <button
                    type="button"
                    className="icon-btn"
                    title={t.scan.mine.keep}
                    aria-label={t.scan.mine.saveIn(item.name)}
                    aria-pressed={mySaved?.some((m) => m.class_id === item.class_id) ?? false}
                    onClick={() => void saveMedicine(item)}
                  >
                    <Icon name="bookmark" size={14} />
                  </button>
                )}
                <button
                  type="button"
                  className="icon-btn"
                  aria-label={t.scan.mine.remove(item.name)}
                  onClick={() => remove(item.key)}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
        <div className="form-actions">
          <button
            type="button"
            className="btn btn-primary"
            disabled={checkable < 2 || checking}
            onClick={check}
          >
            {checking && <span className="btn-spinner" aria-hidden="true" />}
            {t.scan.mine.check}
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            disabled={checkable < 1 || checkable > 10}
            onClick={() => setSaving(true)}
            title={t.scan.mine.saveHint}
          >
            <Icon name="file" size={16} />
            {t.scan.mine.saveReport(kept.length)}
          </button>
        </div>
        {report && <InteractionReport report={report} />}
      </section>

      <ChatPanel pills={mine} kept={kept} onKeep={keep} />

      {saving && (
        <SaveReportDialog
          pills={classIds}
          ingredients={boxIngredients}
          answers={kept}
          defaultTitle={mine.map((item) => item.name).slice(0, 3).join(" + ")}
          onClose={() => setSaving(false)}
        />
      )}
    </>
  );
}
