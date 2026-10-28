import { useEffect, useRef, useState, type DragEvent } from "react";
import {
  api,
  errorMessage,
  type DemoImage,
  type InteractionPair,
  type InteractionsResult,
  type PillCandidate,
  type ScannedPill,
  type ScanResult,
  type Severity,
  type Source,
} from "../api";
import { Alert, Badge, Spinner, staggerIndex } from "../components";
import { useDebounced, useQuery } from "../hooks";
import { Icon, type IconName } from "../icons";

// Safety texts are shown exactly as the project rules write them (CLAUDE.md, D53).
const DISCLAIMER = "Information uniquement, consultez un pharmacien ou un médecin.";
const ACCEPTED_TYPES = "image/jpeg,image/png,image/webp";
const MAX_BYTES = 10 * 1024 * 1024;
const LIVE_PAUSE_MS = 400; // between two live frames, after the previous answer arrived

type Mode = "upload" | "webcam" | "demo";

const MODES: { mode: Mode; label: string; icon: IconName }[] = [
  { mode: "upload", label: "Upload a photo", icon: "upload" },
  { mode: "webcam", label: "Webcam", icon: "camera" },
  { mode: "demo", label: "Demo photos", icon: "image" },
];

/** A pill the user has confirmed (or picked by hand), ready for the interaction check. */
interface Confirmed {
  key: string;
  class_id: number;
  name: string;
  generic_name: string;
  active_ingredients: string[];
  probability: number | null; // null when chosen by search rather than from the scan
}

interface Photo {
  url: string; // object URL of exactly what was sent to the scan
  name: string;
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
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

/** The photo with one numbered box per detected pill, drawn in an SVG on top of it. */
function Overlay({
  alt,
  pills,
  live = false,
}: {
  alt: string;
  pills: ScannedPill[];
  live?: boolean;
}) {
  return (
    <svg className={live ? "scan-overlay scan-overlay-live" : "scan-overlay"} aria-hidden={live}>
      {!live && <title>{alt}</title>}
      {pills.map((pill, index) => {
        const box = pill.box_normalized;
        const top = pill.candidates[0];
        const tone = pill.needs_confirmation ? "warn" : "ok";
        return (
          <g key={index} className={`scan-box scan-box-${tone}`}>
            <rect
              x={`${box.x1 * 100}%`}
              y={`${box.y1 * 100}%`}
              width={`${(box.x2 - box.x1) * 100}%`}
              height={`${(box.y2 - box.y1) * 100}%`}
              rx="6"
            />
            <text x={`${box.x1 * 100}%`} y={`${box.y1 * 100}%`} dx="4" dy="-8">
              {index + 1}. {top ? `${top.name} ${percent(top.probability)}` : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function UploadPanel({ onImage }: { onImage: (file: Blob, name: string) => void }) {
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
        className={dragging ? "dropzone dropzone-active" : "dropzone"}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={drop}
      >
        <Icon name="upload" size={28} />
        <strong>Drop a photo here, or click to choose one</strong>
        <span className="muted">JPEG, PNG or WebP, up to 10 MB. One or several pills on a plain background.</span>
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

/** Live camera: preview with boxes refreshed about once a second, and a capture button. */
function WebcamPanel({ onImage }: { onImage: (file: Blob, name: string) => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [live, setLive] = useState(true);
  const [preview, setPreview] = useState<ScanResult | null>(null);

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
    } catch (failure) {
      setProblem(`The camera is not available: ${errorMessage(failure)}`);
    }
  };

  // Attach the stream to the <video>, and stop the camera when leaving the panel.
  useEffect(() => {
    if (video.current && stream) video.current.srcObject = stream;
    return () => stream?.getTracks().forEach((track) => track.stop());
  }, [stream]);

  // Live predictions: one frame at a time, the next only after the previous answer.
  useEffect(() => {
    if (!stream || !live) {
      setPreview(null);
      return;
    }
    const controller = new AbortController();
    let timer = 0;
    const tick = async () => {
      const element = video.current;
      if (controller.signal.aborted) return;
      if (element && element.videoWidth > 0) {
        try {
          const frame = await grabFrame(element, 960);
          const result = await api.scan(frame, "live.jpg", controller.signal);
          if (!controller.signal.aborted) setPreview(result);
        } catch (failure) {
          if (controller.signal.aborted) return;
          setProblem(errorMessage(failure));
          setLive(false);
          return;
        }
      }
      timer = window.setTimeout(tick, LIVE_PAUSE_MS);
    };
    void tick();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [stream, live]);

  const capture = async () => {
    if (!video.current) return;
    try {
      onImage(await grabFrame(video.current, 1920), "webcam.jpg");
    } catch (failure) {
      setProblem(errorMessage(failure));
    }
  };

  return (
    <>
      {problem && <Alert>{problem}</Alert>}
      {!stream ? (
        <div className="dropzone">
          <Icon name="camera" size={28} />
          <strong>Use your camera</strong>
          <span className="muted">The browser asks for permission. Nothing is recorded or stored.</span>
          <button type="button" className="btn btn-primary" onClick={start}>
            Start the camera
          </button>
        </div>
      ) : (
        <div className="webcam">
          <div className="scan-stage">
            <video ref={video} autoPlay playsInline muted />
            {preview && <Overlay alt="Live predictions" pills={preview.pills} live />}
          </div>
          <div className="inline-actions">
            <button type="button" className="btn btn-primary" onClick={capture}>
              <Icon name="camera" size={16} />
              Capture and identify
            </button>
            <label className="check">
              <input type="checkbox" checked={live} onChange={(event) => setLive(event.target.checked)} />
              Live predictions
            </label>
            {preview && (
              <span className="muted" aria-live="polite">
                {preview.pills.length} pill(s), {Math.round(preview.inference_ms)} ms
              </span>
            )}
          </div>
        </div>
      )}
    </>
  );
}

function DemoPanel({ onImage }: { onImage: (file: Blob, name: string) => void }) {
  const gallery = useQuery((signal) => api.demoImages(signal), []);
  const [loading, setLoading] = useState<number | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  const pick = async (image: DemoImage) => {
    setLoading(image.id);
    setProblem(null);
    try {
      onImage(await api.demoImage(image), `demo-${image.id}.jpg`);
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
      <p className="muted">{gallery.data.note} Click a photo to identify it.</p>
      <div className="gallery">
        {gallery.data.items.map((image, index) => (
          <button
            type="button"
            key={image.id}
            className="gallery-item"
            style={staggerIndex(index)}
            onClick={() => pick(image)}
            disabled={loading !== null}
            data-testid={`demo-${image.id}`}
          >
            <img src={image.thumbnail_url} alt={`C3PI test photo ${image.id + 1}`} loading="lazy" />
            <span className="gallery-caption">
              {loading === image.id ? "Loading…" : `Shows: ${image.expected_name || `class ${image.class_id}`}`}
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
                <span className="prob" title="Model probability">
                  <span className="prob-bar">
                    <span style={{ width: `${candidate.probability * 100}%` }} />
                  </span>
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

export default function ScanPage() {
  const [mode, setMode] = useState<Mode>("upload");
  const [photo, setPhoto] = useState<Photo | null>(null);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mine, setMine] = useState<Confirmed[]>([]);
  const [report, setReport] = useState<InteractionsResult | null>(null);
  const [checking, setChecking] = useState(false);
  const resultRef = useRef<HTMLElement>(null);

  // Release the previous photo's memory when it is replaced or the page is left.
  useEffect(() => () => {
    if (photo) URL.revokeObjectURL(photo.url);
  }, [photo]);

  const identify = async (image: Blob, name: string) => {
    setScanning(true);
    setError(null);
    setResult(null);
    setPhoto({ url: URL.createObjectURL(image), name });
    try {
      setResult(await api.scan(image, name));
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
        class_id: choice.class_id,
        name: choice.name,
        generic_name: choice.generic_name,
        active_ingredients: choice.active_ingredients,
        probability: Number.isNaN(choice.probability) ? null : choice.probability,
      },
    ]);
    setReport(null);
  };

  const remove = (key: string) => {
    setMine((current) => current.filter((item) => item.key !== key));
    setReport(null);
  };

  // The same product confirmed twice (front and back of one pill) is checked once.
  const classIds = [...new Set(mine.map((item) => item.class_id))];

  const check = async () => {
    setChecking(true);
    setError(null);
    try {
      setReport(await api.interactions(classIds));
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
              Identify pills from a photo, confirm them, then check their active ingredients for
              interactions.
            </p>
          </div>
        </div>
      </header>

      <Disclaimer />

      <section className="card scan-input" aria-label="Photo">
        <div className="segmented" role="tablist" aria-label="Photo source">
          {MODES.map((item) => (
            <button
              type="button"
              role="tab"
              key={item.mode}
              aria-selected={mode === item.mode}
              className={mode === item.mode ? "segment segment-on" : "segment"}
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
        <section className="scan-result" ref={resultRef} aria-label="Result" aria-busy={scanning}>
          <div className="card scan-photo">
            <div className="scan-stage">
              <img src={photo.url} alt={`Photo sent for identification: ${photo.name}`} />
              {result && <Overlay alt="Detected pills" pills={result.pills} />}
              {scanning && (
                <div className="scan-busy">
                  <Spinner label="Identifying the pills" />
                </div>
              )}
            </div>
            {result && (
              <p className="muted scan-meta">
                {result.pills.length} pill(s) found · {result.image_width} × {result.image_height}{" "}
                px · {Math.round(result.inference_ms)} ms (ONNX Runtime) · confirmation below{" "}
                {percent(result.confirm_threshold)}
              </p>
            )}
          </div>
          <div className="scan-pills">
            {result?.message && <Alert kind="info">{result.message}</Alert>}
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
            {mine.length} confirmed, {classIds.length} different
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
            disabled={classIds.length < 2 || checking}
            onClick={check}
          >
            {checking && <span className="btn-spinner" aria-hidden="true" />}
            Check interactions
          </button>
        </div>
        {report && <InteractionReport report={report} />}
      </section>
    </>
  );
}
