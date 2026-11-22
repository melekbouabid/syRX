// Two small SVG charts for the admin Usage page: stacked daily columns and a daily line.
// Colours are CSS variables of the `.viz` scope (styles.css), validated for colour-blind
// readers in light and dark mode; every chart has a legend, a hover tooltip and a table view
// on the page, so no value is told by colour alone.

import { useEffect, useRef, useState, type RefObject } from "react";
import { locale, messages } from "../i18n";
import { formatNumber } from "./index";

export interface Series {
  key: string;
  label: string;
  /** CSS colour, e.g. var(--series-1). */
  color: string;
}

/** One day of data: `day` (YYYY-MM-DD) and a number per series key. */
export type DayRow = { day: string } & object;

function valueOf(row: DayRow, key: string): number {
  return Number((row as Record<string, unknown>)[key] ?? 0);
}

const HEIGHT = 240;
const PAD = { top: 12, right: 12, bottom: 28, left: 40 };
const GAP = 2; // surface gap between stacked segments
const MAX_BAR = 24;

function useWidth(ref: RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(640);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.round(entry.contentRect.width)));
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref]);
  return width;
}

/** 0 and up to four clean steps (1, 2 or 5 times a power of ten) that reach the maximum. */
export function niceTicks(max: number): number[] {
  if (max <= 0) return [0, 1];
  const rough = max / 4;
  const power = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= rough) ?? power * 10;
  const ticks = [];
  for (let value = 0; value < max + step; value += step) ticks.push(Math.max(0, Math.round(value)));
  return [...new Set(ticks)];
}

export function shortDay(iso: string): string {
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString(locale(), {
    day: "numeric",
    month: "short",
    timeZone: "UTC",
  });
}

function longDay(iso: string): string {
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString(locale(), {
    weekday: "long",
    day: "numeric",
    month: "long",
    timeZone: "UTC",
  });
}

/** Indexes of the days that get a label under the axis: about six, first and last included. */
function labelled(count: number): Set<number> {
  const step = Math.max(1, Math.ceil(count / 6));
  const shown = new Set<number>();
  for (let i = count - 1; i >= 0; i -= step) shown.add(i);
  return shown;
}

/** A path for a column whose top corners are rounded and whose base is square. */
function roundedTop(x: number, y: number, w: number, h: number, r = 4): string {
  const radius = Math.min(r, w / 2, h);
  return `M${x},${y + h}V${y + radius}Q${x},${y} ${x + radius},${y}H${x + w - radius}Q${x + w},${y} ${x + w},${y + radius}V${y + h}Z`;
}

function Axis({ ticks, scale, width }: { ticks: number[]; scale: (v: number) => number; width: number }) {
  return (
    <g className="viz-axis">
      {ticks.map((tick) => (
        <g key={tick}>
          <line x1={PAD.left} x2={width - PAD.right} y1={scale(tick)} y2={scale(tick)} />
          <text x={PAD.left - 8} y={scale(tick)} dy="0.32em" textAnchor="end">
            {formatNumber(tick)}
          </text>
        </g>
      ))}
    </g>
  );
}

function DayLabels({ rows, x, band }: { rows: DayRow[]; x: (i: number) => number; band: number }) {
  const shown = labelled(rows.length);
  return (
    <g className="viz-axis">
      {rows.map((row, i) =>
        shown.has(i) ? (
          <text key={row.day} x={x(i) + band / 2} y={HEIGHT - 8} textAnchor="middle">
            {shortDay(row.day)}
          </text>
        ) : null,
      )}
    </g>
  );
}

interface TooltipProps {
  left: number;
  width: number;
  title: string;
  lines: { label: string; value: number; color?: string }[];
}

function Tooltip({ left, width, title, lines }: TooltipProps) {
  const flip = left > width * 0.6;
  return (
    <div className="viz-tooltip" style={flip ? { right: width - left + 12 } : { left: left + 12 }} role="status">
      <strong>{title}</strong>
      {lines.map((line) => (
        <span key={line.label} className="viz-tooltip-row">
          {line.color && <i style={{ background: line.color }} aria-hidden="true" />}
          {line.label}
          <b>{formatNumber(line.value)}</b>
        </span>
      ))}
    </div>
  );
}

export function Legend({ series }: { series: Series[] }) {
  return (
    <ul className="viz-legend">
      {series.map((s) => (
        <li key={s.key}>
          <i style={{ background: s.color }} aria-hidden="true" />
          {s.label}
        </li>
      ))}
    </ul>
  );
}

/** One column per day, one stacked segment per series, a tooltip on the hovered day. */
export function StackedColumns({ rows, series, label }: { rows: DayRow[]; series: Series[]; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const width = useWidth(ref);
  const [active, setActive] = useState<number | null>(null);
  const totals = rows.map((row) => series.reduce((sum, s) => sum + valueOf(row, s.key), 0));
  const ticks = niceTicks(Math.max(...totals, 0));
  const top = ticks[ticks.length - 1];
  const plotW = width - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const band = plotW / Math.max(rows.length, 1);
  const bar = Math.min(MAX_BAR, band * 0.64);
  const scale = (v: number) => PAD.top + plotH - (v / top) * plotH;
  const x = (i: number) => PAD.left + i * band;

  return (
    <div className="viz-plot" ref={ref} onMouseLeave={() => setActive(null)}>
      <svg width={width} height={HEIGHT} role="img" aria-label={label}>
        <Axis ticks={ticks} scale={scale} width={width} />
        {rows.map((row, i) => {
          let base = 0;
          const present = series.filter((s) => valueOf(row, s.key) > 0);
          return (
            <g key={row.day} opacity={active === null || active === i ? 1 : 0.45}>
              {present.map((s, j) => {
                const value = valueOf(row, s.key);
                const y0 = scale(base);
                base += value;
                const y1 = scale(base);
                const last = j === present.length - 1;
                const h = Math.max(1, y0 - y1 - (last ? 0 : GAP));
                const left = x(i) + (band - bar) / 2;
                return last ? (
                  <path key={s.key} d={roundedTop(left, y0 - h, bar, h)} fill={s.color} />
                ) : (
                  <rect key={s.key} x={left} y={y0 - h} width={bar} height={h} fill={s.color} />
                );
              })}
              {/* The whole band is the hover target, wider than the column. */}
              <rect
                x={x(i)}
                y={PAD.top}
                width={band}
                height={plotH}
                fill="transparent"
                onMouseEnter={() => setActive(i)}
                onFocus={() => setActive(i)}
                tabIndex={0}
                aria-label={messages().charts.actions(longDay(row.day), totals[i])}
              />
            </g>
          );
        })}
        <line className="viz-baseline" x1={PAD.left} x2={width - PAD.right} y1={scale(0)} y2={scale(0)} />
        <DayLabels rows={rows} x={x} band={band} />
      </svg>
      {active !== null && rows[active] && (
        <Tooltip
          left={x(active) + band / 2}
          width={width}
          title={`${longDay(rows[active].day)} · ${formatNumber(totals[active])}`}
          lines={series.map((s) => ({ label: s.label, value: valueOf(rows[active], s.key), color: s.color }))}
        />
      )}
    </div>
  );
}

/** One value per day as a 2px line, with a crosshair and a tooltip on the hovered day. */
export function DailyLine({
  rows,
  field,
  name,
  color,
  label,
}: {
  rows: DayRow[];
  field: string;
  name: string;
  color: string;
  label: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const width = useWidth(ref);
  const [active, setActive] = useState<number | null>(null);
  const values = rows.map((row) => valueOf(row, field));
  const ticks = niceTicks(Math.max(...values, 0));
  const top = ticks[ticks.length - 1];
  const plotW = width - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const band = plotW / Math.max(rows.length, 1);
  const scale = (v: number) => PAD.top + plotH - (v / top) * plotH;
  const x = (i: number) => PAD.left + i * band;
  const cx = (i: number) => x(i) + band / 2;
  const path = values.map((v, i) => `${i ? "L" : "M"}${cx(i)},${scale(v)}`).join("");

  return (
    <div className="viz-plot" ref={ref} onMouseLeave={() => setActive(null)}>
      <svg width={width} height={HEIGHT} role="img" aria-label={label}>
        <Axis ticks={ticks} scale={scale} width={width} />
        <line className="viz-baseline" x1={PAD.left} x2={width - PAD.right} y1={scale(0)} y2={scale(0)} />
        {active !== null && (
          <line className="viz-crosshair" x1={cx(active)} x2={cx(active)} y1={PAD.top} y2={scale(0)} />
        )}
        <path d={path} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {values.length > 0 && (
          <circle
            cx={cx(values.length - 1)}
            cy={scale(values[values.length - 1])}
            r={4}
            fill={color}
            className="viz-dot"
          />
        )}
        {active !== null && (
          <circle cx={cx(active)} cy={scale(values[active])} r={5} fill={color} className="viz-dot" />
        )}
        {rows.map((row, i) => (
          <rect
            key={row.day}
            x={x(i)}
            y={PAD.top}
            width={band}
            height={plotH}
            fill="transparent"
            onMouseEnter={() => setActive(i)}
            onFocus={() => setActive(i)}
            tabIndex={0}
            aria-label={`${longDay(row.day)}: ${values[i]} ${name}`}
          />
        ))}
        <DayLabels rows={rows} x={x} band={band} />
      </svg>
      {active !== null && rows[active] && (
        <Tooltip
          left={cx(active)}
          width={width}
          title={longDay(rows[active].day)}
          lines={[{ label: name, value: values[active], color }]}
        />
      )}
    </div>
  );
}
