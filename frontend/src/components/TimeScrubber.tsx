import { useMemo } from "react";

import { useAppStore } from "../store/useAppStore";

export function TimeScrubber() {
  const { horizons, activeHorizon, setHorizon } = useAppStore((state) => ({
    horizons: state.horizons,
    activeHorizon: state.activeHorizon,
    setHorizon: state.setHorizon,
  }));

  if (!horizons.length || activeHorizon == null) {
    return null;
  }

  const min = horizons[0];
  const max = horizons[horizons.length - 1];
  const marks = useMemo(() => horizons.map((h) => ({ value: h, label: `${h}h` })), [horizons]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-slate-600">
        <span>Forecast horizon</span>
        <span className="font-medium text-slate-800">{activeHorizon}h</span>
      </div>
      <input
        className="w-full"
        type="range"
        min={min}
        max={max}
        step={1}
        value={activeHorizon}
        onChange={(event) => setHorizon(Number(event.target.value))}
      />
      <div className="flex justify-between text-[10px] text-slate-500">
        {marks.map((mark) => (
          <span key={mark.value}>{mark.label}</span>
        ))}
      </div>
    </div>
  );
}

