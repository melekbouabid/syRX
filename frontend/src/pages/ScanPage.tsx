import { useEffect, useRef, useState, type DragEvent } from "react";
import { Link } from "react-router-dom";
import {
  api,
  errorMessage,
  type BoxReading,
  type DemoImage,
  type InteractionPair,
  type InteractionsResult,
  type PillCandidate,
  type ScannedPill,
  type ScanResult,
  type Severity,
  type Source,
} from "../services/api";
import { Alert, Badge, Spinner, staggerIndex } from "../components";
import { useDebounced, useQuery } from "../hooks";
import { Icon, type IconName } from "../components/icons";
import { LatencyReadout, ModelCard, ProbBar, ProbabilityBars, ScanCanvas } from "../components/vision";
import { percent, startLiveLoop, type Latency } from "../lib/vision";
import { showDriftBanner } from "../lib/drift";
import ChatPanel from "./ChatPanel";

// Safety texts are shown exactly as the project rules write them (docs/project_rules.md, D53).
const DISCLAIMER = "Information uniquement, consultez un pharmacien ou un médecin.";
const ACCEPTED_TYPES = "image/jpeg,image/png,image/webp";
const MAX_BYTES = 10 * 1024 * 1024;
const LIVE_PAUSE_MS = 400; // between two live frames, after the previous answer arrived
const LIVE_WIDTH = 960; // live frames are scaled down to this width before upload

type Mode = "upload" | "webcam" | "demo";

const MODES: { mode: Mode; label: string; icon: IconName }[] = [
  { mode: "upload", label: "Upload or drop a photo", icon: "upload" },
  { mode: "demo", label: "Demo gallery", icon: "image" },
  { mode: "webcam", label: "Webcam (live)", icon: "camera" },
];

const WHY: Record<string, string> = {
  major: "Major interactions",
  confirm: "Needs confirmation",
  "two-pills": "Two pills",
  "no-ingredient": "No known ingredient",
};

const METHOD: Record<string, string> = {
  brand: "brand table",
  printed_name: "name printed on the box",
  rxnav: "RxNorm (RxNav)",
};

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
  return (
    <article className={`card pill-card${box.recognised ? "" : " pill-card-warn"}`} data-testid="box-card">
      <header className="pill-card-head">
        <span className="pill-number">
          <Icon name="pill" size={16} />
        </span>
        <div>
          <h3>{box.brand ?? (box.recognised ? "Medicine box" : "Box not recognised")}</h3>
          <p className="muted">
            Medicine box{box.brand_country ? ` · brand listed in ${box.brand_country}` : ""} ·
            confidence {percent(box.confidence)}
          </p>
        </div>
        <Badge tone="warn">{box.recognised ? "Check the box" : "Not recognised"}</Badge>
      </header>
      <p className="pill-note">
        <Icon name="alert" size={16} /> Read from the printed text, a suggestion only: compare it
        with the box and its leaflet.
      </p>
      {box.ingredients.length > 0 ? (
        <ul className="m-0 grid list-none gap-2 p-0" aria-label="Active ingredients found">
          {box.ingredients.map((item) => (
            <li key={item.name} className="rounded-xl border border-line bg-surface-2 p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="cell-title">{item.name}</span>
                <span
                  className="flex w-32 flex-none items-center gap-2 text-[0.88rem] font-semibold tabular-nums"
                  title="OCR confidence x similarity of the words"
                >
                  <ProbBar value={item.confidence} />
                  {percent(item.confidence)}
                </span>
              </div>
              <p className="cell-sub m-0">
                Read: “{item.matched_text}” · from the {METHOD[item.method] ?? item.method}
                {item.in_interaction_table ? "" : " · not in the interaction table"}
              </p>
              <p className="cell-sub m-0">
                Source:{" "}
                <a href={item.url} target="_blank" rel="noreferrer">
                  {item.source}
                </a>{" "}
                · {item.reference}
              </p>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted m-0">
          No brand and no active ingredient of this database was read: nothing is suggested.
        </p>
      )}
      <details className="text-sm">
        <summary className="cursor-pointer font-semibold">Text read ({box.text.length} lines)</summary>
        <p className="muted m-0 mt-2 whitespace-pre-line" data-testid="box-text">
          {box.text.join("\n") || "no text"}
        </p>
      </details>
      <p className="muted m-0 text-xs">RxNav fallback: {box.rxnav}</p>
      {box.recognised && (
        <div className="form-actions">
          <button type="button" className="btn btn-primary" disabled={added} onClick={onAdd}>
            {added ? "Added to my pills" : "Add to my pills"}
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

function Disclaimer({ text = DISCLAIMER }: { text?: string }) {
  return (
    <p className="disclaimer">
      <Icon name="shield" size={16} />
      {text}
    </p>
  );
}

function SourceList({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;
  return (
    <div className="cited-sources">
      <h4>Sources</h4>
      <ul>
        {sources.map((source) => (
          <li key={source.name}>
            <a href={source.url} target="_blank" rel="noreferrer">
              {source.title}
            </a>{" "}
            <span className="muted">({source.license})</span>
            {source.citation && <span className="cell-sub">{source.citation}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Drag and drop (or click) a photo; checked here before the API checks it again. */
function UploadPanel({ onImage }: { onImage: OnImage }) {
  const [dragging, setDragging] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const take = (file: File | undefined) => {
    if (!file) return;
    if (!ACCEPTED_TYPES.split(",").includes(file.type)) {
      setProblem("Choose a JPEG, PNG or WebP image.");
      return;
    }
    if (file.size > MAX_BYTES) {
      setProblem("The image is larger than 10 MB.");
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
          {dragging ? "Drop it: the pills are identified at once" : "Drag and drop a photo here, or click to choose one"}
        </strong>
        <span className="text-sm text-muted">
          JPEG, PNG or WebP, up to 10 MB. One or several pills on a plain background.
        </span>
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
      (blob) => (blob ? resolve(blob) : reject(new Error("Cannot read the camera image."))),
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
  if (!frame) {
    return <p className="m-0 text-sm text-muted">Waiting for the first answer of the API…</p>;
  }
  const { result } = frame;
  return (
    <div className="grid content-start gap-3" aria-live="polite">
      <LatencyReadout latency={frame.latency} />
      <p className="m-0 text-sm text-muted">
        Frame {frame.count} · {result.pills.length} pill(s) · one request at a time, the next
        frame after the answer
      </p>
      {result.pills.length === 0 && (
        <p className="m-0 text-sm text-muted">{result.message ?? "No pill in view."}</p>
      )}
      {result.pills.slice(0, 3).map((pill, index) => (
        <div key={index} className="grid gap-2 rounded-xl border border-line bg-surface p-3">
          <div className="flex items-center gap-2">
            <span className="pill-number">{index + 1}</span>
            <span className="min-w-0 flex-1 truncate font-semibold text-heading">
              {pill.candidates[0]?.name}
            </span>
            {pill.needs_confirmation ? (
              <Badge tone="warn">Please confirm</Badge>
            ) : (
              <Badge tone="active">High confidence</Badge>
            )}
          </div>
          <ProbabilityBars candidates={pill.candidates} />
          <span className="text-xs text-muted">
            Detection confidence {percent(pill.detection_confidence)} · a suggestion, never a
            certainty
          </span>
        </div>
      ))}
    </div>
  );
}

/** Live camera: continuous inference with boxes and bars, and a capture button. */
function WebcamPanel({ onImage }: { onImage: OnImage }) {
  const video = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [live, setLive] = useState(true);
  const [frame, setFrame] = useState<LiveFrame | null>(null);

  const start = async () => {
    setProblem(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setProblem("This browser cannot use a camera here (it needs HTTPS or localhost).");
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
      setProblem(`The camera is not available: ${errorMessage(failure)}`);
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
          <strong className="text-heading">Use your camera</strong>
          <span className="text-sm text-muted">
            Each frame is identified continuously (about twice a second, one request at a time).
            The browser asks for permission; nothing is recorded or stored.
          </span>
          <button type="button" className="btn btn-primary" onClick={start}>
            Start the camera
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
                  label="Live predictions"
                />
              )}
              {live && (
                <span className="absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-full bg-black/60 px-2.5 py-1 text-xs font-bold tracking-wide text-white">
                  <span className="size-2 animate-pulse rounded-full bg-red-500" aria-hidden="true" />
                  LIVE
                </span>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <button type="button" className="btn btn-primary" onClick={capture}>
                <Icon name="camera" size={16} />
                Capture and identify
              </button>
              <label className="check">
                <input type="checkbox" checked={live} onChange={(event) => setLive(event.target.checked)} />
                Live predictions
              </label>
              <button type="button" className="btn btn-ghost btn-sm" onClick={stop}>
                Stop the camera
              </button>
            </div>
          </div>
          {live ? (
            <LivePanel frame={frame} />
          ) : (
            <p className="m-0 text-sm text-muted">Live predictions are paused.</p>
          )}
        </div>
      )}
    </>
  );
}

function DemoPanel({ onImage }: { onImage: OnImage }) {
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
  if (!gallery.data) return <Spinner label="Loading the demo photos" />;
  if (gallery.data.items.length === 0) {
    return <Alert kind="info">No demo photo is available on this server (C3PI images not mounted).</Alert>;
  }
  return (
    <>
      {problem && <Alert>{problem}</Alert>}
      <p className="mb-3 mt-0 text-sm text-muted">
        {gallery.data.note} One click identifies a photo.
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
                alt={`C3PI test photo ${image.id + 1}`}
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
              {loading === image.id ? "Loading…" : `Shows: ${image.expected_name || `class ${image.class_id}`}`}
              {WHY[image.why] && <span className="block text-xs text-muted">{WHY[image.why]}</span>}
            </span>
          </button>
        ))}
      </div>
    </>
  );
}

/** Search the 150 classes by name, ingredient or imprint, to correct an identification. */
function PillSearch({ onPick }: { onPick: (pill: PillCandidate) => void }) {
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
          placeholder="Not in the list? Search a name, ingredient or imprint"
          aria-label="Search another pill"
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
                  {[pill.imprint && `imprint ${pill.imprint}`, pill.color, pill.shape]
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              </button>
            </li>
          ))}
          {found.data.items.length === 0 && <li className="muted">No pill matches “{q}”.</li>}
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
            Most likely, {percent(pill.candidates[0].probability)} · detection{" "}
            {percent(pill.detection_confidence)}
          </p>
        </div>
        {pill.needs_confirmation ? (
          <Badge tone="warn">Please confirm</Badge>
        ) : (
          <Badge tone="active">High confidence</Badge>
        )}
      </header>
      {pill.needs_confirmation && (
        <p className="pill-note">
          <Icon name="alert" size={16} /> The model is not sure. Compare the photo with the
          imprint and colour, pick the right pill below, or search for it.
        </p>
      )}
      <fieldset className="candidates">
        <legend className="sr-only">Possible pills for pill {index + 1}</legend>
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
                    : "no active ingredient known in this database"}
                </span>
              </span>
              {searched ? (
                <span className="muted">chosen by search</span>
              ) : (
                <span
                  className="flex w-32 flex-none items-center gap-2 text-[0.88rem] font-semibold tabular-nums"
                  title="Model probability"
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
          {confirmed ? "Added to my pills" : "Confirm this pill"}
        </button>
      </div>
    </article>
  );
}

const SEVERITY_TONE: Record<Severity, string> = {
  Major: "danger",
  Moderate: "warn",
  Minor: "info",
  Unknown: "neutral",
};

const SEVERITY_TEXT: Record<Severity, string> = {
  Major: "Major",
  Moderate: "Moderate",
  Minor: "Minor",
  Unknown: "Unknown severity (DDInter lists the interaction without grading it)",
};

function PairCard({ pair, index }: { pair: InteractionPair; index: number }) {
  const tone = pair.highest_severity ? SEVERITY_TONE[pair.highest_severity] : "none";
  return (
    <article className={`card pair-card pair-${tone}`} style={staggerIndex(index)}>
      <header className="pair-head">
        <h3>
          {pair.item_a.label} <span className="muted">+</span> {pair.item_b.label}
        </h3>
        {pair.highest_severity && (
          <Badge tone={SEVERITY_TONE[pair.highest_severity]}>{pair.highest_severity}</Badge>
        )}
      </header>
      {pair.message && <p className="pair-message">{pair.message}</p>}
      {pair.interactions.length > 0 && (
        <ul className="interaction-list">
          {pair.interactions.map((found) => (
            <li key={`${found.ingredient_a}-${found.ingredient_b}`}>
              <strong>
                {found.ingredient_a} + {found.ingredient_b}
              </strong>
              : {SEVERITY_TEXT[found.severity]}. {found.description}{" "}
              <a href={found.source.url} target="_blank" rel="noreferrer">
                Source: {found.source.title}
              </a>
            </li>
          ))}
        </ul>
      )}
      {pair.shared_ingredients.length > 0 && (
        <p className="pill-note">
          <Icon name="alert" size={16} /> Same active ingredient in both:{" "}
          {pair.shared_ingredients.join(", ")}.
        </p>
      )}
    </article>
  );
}

function InteractionReport({ report }: { report: InteractionsResult }) {
  const counts = report.pairs.reduce<Record<string, number>>((total, pair) => {
    const key = pair.highest_severity ?? "none";
    total[key] = (total[key] ?? 0) + 1;
    return total;
  }, {});
  return (
    <section className="report" aria-live="polite">
      <div className={`report-summary report-${report.highest_severity ? SEVERITY_TONE[report.highest_severity] : "none"}`}>
        <p className="report-message">{report.message}</p>
        <span className="badge-row">
          {(["Major", "Moderate", "Minor", "Unknown"] as Severity[]).map(
            (severity) =>
              counts[severity] && (
                <Badge key={severity} tone={SEVERITY_TONE[severity]}>
                  {counts[severity]} {severity}
                </Badge>
              ),
          )}
        </span>
      </div>
      {report.pairs.map((pair, index) => (
        <PairCard key={`${pair.item_a.label}|${pair.item_b.label}`} pair={pair} index={index} />
      ))}
      <SourceList sources={report.sources} />
      <Disclaimer text={report.disclaimer} />
    </section>
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
  const [view, setView] = useState<View>("photo");
  const shown = view === "gradcam" && photo.gradcamUrl ? photo.gradcamUrl : photo.url;

  return (
    <div className="card sticky top-[84px] grid gap-3 p-3" data-testid="result-stage">
      {photo.gradcamUrl && (
        <div className="flex gap-1 rounded-xl bg-surface-2 p-1" role="tablist" aria-label="Picture shown">
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
              {item === "photo" ? "Photo and boxes" : "Grad-CAM"}
            </button>
          ))}
        </div>
      )}
      <div className="relative overflow-hidden rounded-xl bg-surface-2 leading-none">
        <img
          src={shown}
          alt={
            view === "gradcam"
              ? `Grad-CAM of the classifier on ${photo.name}`
              : `Photo sent for identification: ${photo.name}`
          }
          className="block h-auto w-full"
        />
        {result && (
          <ScanCanvas
            pills={result.pills}
            width={result.image_width}
            height={result.image_height}
            label="Detected pills"
          />
        )}
        {scanning && (
          <div className="absolute inset-0 grid place-items-center bg-surface/60" data-testid="scan-busy">
            <Spinner label="Identifying the pills" />
          </div>
        )}
      </div>
      {view === "gradcam" && photo.gradcamUrl && (
        <p className="m-0 text-sm text-muted">
          Grad-CAM of the served classifier for its top-1 class: red marks the parts of the crop
          that weigh most in that class, blue the least. Computed offline with the PyTorch weights
          of the same version (the API runs ONNX only). It explains the suggestion; it does not
          make it more certain.
        </p>
      )}
      {!photo.gradcamUrl && result && result.kind !== "box" && result.kind !== "unknown" && (
        <p className="m-0 text-sm text-muted" data-testid="gradcam-note">
          Grad-CAM is available only for the demo gallery photos: it needs the PyTorch model,
          so it is precomputed for them (the API runs ONNX models only).
        </p>
      )}
      {result && (
        <>
          <p className="m-0 text-sm text-muted" data-testid="scan-kind">
            {result.kind === "box"
              ? `Medicine box: printed text read${
                  result.timings.ocr_ms != null ? ` in ${Math.round(result.timings.ocr_ms)} ms` : ""
                }, no pill class is given to a box`
              : result.kind === "unknown"
                ? "Neither a pill nor a medicine box: nothing is identified"
                : `${result.pills.length} pill(s) found · ${result.image_width} × ${result.image_height} px · boxes coloured by class, dashed below ${percent(result.confirm_threshold)} (please confirm)`}
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
  const drift = useQuery((signal) => api.drift(signal), [scans]);
  const report = drift.data;
  if (!report || !showDriftBanner(report.status)) return null;
  return (
    <Alert>
      Data drift: the recent photos differ from the reference photos ({report.drifted.join(", ")}).
      Identification is less reliable; take the photo in good light, sharp, one pill on a plain
      background. <Link to="/monitoring">See the monitoring page</Link>.
    </Alert>
  );
}

export default function ScanPage() {
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
            <h1>Scanner</h1>
            <p className="muted">
              Identify pills from a photo, confirm them, check their active ingredients for
              interactions, then ask questions answered from cited sources.
            </p>
          </div>
        </div>
      </header>

      <Disclaimer />

      <ModelCard />
      <DriftBanner scans={scans} />

      <section className="card mb-5" aria-label="Photo">
        <div className="mb-4 flex flex-wrap gap-1 rounded-xl bg-surface-2 p-1" role="tablist" aria-label="Photo source">
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
              {item.label}
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
          aria-label="Result"
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

      <section className="card my-pills" aria-label="My pills">
        <header className="pill-card-head">
          <h2>My pills</h2>
          <span className="muted">
            {mine.length} added · {classIds.length} different pill(s)
            {boxIngredients.length ? ` · ${boxIngredients.length} ingredient(s) from boxes` : ""}
          </span>
        </header>
        {mine.length === 0 ? (
          <p className="muted">
            Confirmed pills appear here. Scan several photos to add more; with two or more you can
            check their interactions.
          </p>
        ) : (
          <ul className="chip-list">
            {mine.map((item) => (
              <li key={item.key} className="chip">
                <span>
                  <strong>{item.name}</strong>
                  <span className="cell-sub">
                    {item.active_ingredients.join(" + ") || "no active ingredient known"}
                    {item.probability !== null ? ` · ${percent(item.probability)}` : " · chosen by search"}
                  </span>
                </span>
                <button
                  type="button"
                  className="icon-btn"
                  aria-label={`Remove ${item.name}`}
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
            Check interactions
          </button>
        </div>
        {report && <InteractionReport report={report} />}
      </section>

      <ChatPanel pills={mine} />
    </>
  );
}
