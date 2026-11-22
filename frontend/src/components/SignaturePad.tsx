import { useEffect, useRef, useState, type PointerEvent } from "react";
import { useT } from "../i18n";

/**
 * Draw a signature with a mouse, a pen or a finger. The drawing is a transparent PNG; the
 * server crops it to the ink and re-encodes it (backend/app/images.py) before storing it.
 */
export function SignaturePad({
  onDone,
  busy,
}: {
  onDone: (png: Blob) => void;
  busy: boolean;
}) {
  const t = useT();
  const canvas = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const last = useRef<{ x: number; y: number } | null>(null);
  const [empty, setEmpty] = useState(true);

  // A canvas sized in device pixels, so the strokes stay sharp on high-density screens.
  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    const ratio = window.devicePixelRatio || 1;
    const { width, height } = element.getBoundingClientRect();
    element.width = Math.round(width * ratio);
    element.height = Math.round(height * ratio);
    const context = element.getContext("2d");
    if (!context) return;
    context.scale(ratio, ratio);
    context.lineCap = "round";
    context.lineJoin = "round";
    context.lineWidth = 2.4;
    context.strokeStyle = "#111827";
  }, []);

  const point = (event: PointerEvent<HTMLCanvasElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    return { x: event.clientX - box.left, y: event.clientY - box.top };
  };

  const start = (event: PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    drawing.current = true;
    last.current = point(event);
  };

  const move = (event: PointerEvent<HTMLCanvasElement>) => {
    if (!drawing.current || !last.current) return;
    const context = event.currentTarget.getContext("2d");
    const next = point(event);
    if (!context) return;
    context.beginPath();
    context.moveTo(last.current.x, last.current.y);
    context.lineTo(next.x, next.y);
    context.stroke();
    last.current = next;
    setEmpty(false);
  };

  const stop = () => {
    drawing.current = false;
    last.current = null;
  };

  const clear = () => {
    const element = canvas.current;
    element?.getContext("2d")?.clearRect(0, 0, element.width, element.height);
    setEmpty(true);
  };

  const save = () => canvas.current?.toBlob((blob) => blob && onDone(blob), "image/png");

  return (
    <div className="signature-pad">
      <canvas
        ref={canvas}
        className="signature-canvas"
        aria-label={t.signature.area}
        onPointerDown={start}
        onPointerMove={move}
        onPointerUp={stop}
        onPointerCancel={stop}
        onPointerLeave={stop}
      />
      <div className="form-actions">
        <button type="button" className="btn btn-ghost btn-sm" onClick={clear} disabled={empty || busy}>
          {t.signature.clear}
        </button>
        <button type="button" className="btn btn-primary btn-sm" onClick={save} disabled={empty || busy}>
          {busy && <span className="btn-spinner" aria-hidden="true" />}
          {t.signature.save}
        </button>
      </div>
    </div>
  );
}
