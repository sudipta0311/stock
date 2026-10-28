import englishCopy from "../copy/en.json";

export interface DonutSegment {
  key: string;
  label: string;
  percent: number;
  color: string;
}

export function AllocationDonut({ segments }: { segments: DonutSegment[] }) {
  const circumference = 2 * Math.PI * 36;
  const segmentLength = (percent: number) =>
    (circumference * Math.max(0, Math.min(percent, 100))) / 100;
  return (
    <figure
      className="flex items-center gap-4"
      aria-label={englishCopy.household_ui.plan_mix_chart}
    >
      <svg
        viewBox="0 0 100 100"
        role="img"
        aria-label={englishCopy.household_ui.plan_mix_by_type}
        className="size-28 -rotate-90 shrink-0"
      >
        <circle
          cx="50"
          cy="50"
          r="36"
          fill="none"
          stroke="currentColor"
          strokeWidth="16"
          className="text-muted"
        />
        {segments.map((segment, index) => {
          const length = segmentLength(segment.percent);
          const offset = segments
            .slice(0, index)
            .reduce(
              (total, previous) => total + segmentLength(previous.percent),
              0,
            );
          const dashOffset = -offset;
          return (
            <circle
              key={segment.key}
              cx="50"
              cy="50"
              r="36"
              fill="none"
              stroke={segment.color}
              strokeWidth="16"
              strokeDasharray={`${length} ${circumference - length}`}
              strokeDashoffset={dashOffset}
            />
          );
        })}
      </svg>
      <figcaption className="grid gap-1 text-xs text-muted-foreground">
        <span>{englishCopy.household_ui.plan_mix_caption}</span>
        {segments.map((segment) => (
          <span key={segment.key}>
            {segment.label} · {segment.percent}%
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
