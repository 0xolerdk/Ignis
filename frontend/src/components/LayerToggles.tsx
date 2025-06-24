import { useAppStore } from "../store/useAppStore";

export function LayerToggles() {
  const { toggles, toggleLayer, setUncertaintyOpacity } = useAppStore((state) => ({
    toggles: state.toggles,
    toggleLayer: state.toggleLayer,
    setUncertaintyOpacity: state.setUncertaintyOpacity,
  }));

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm">
        <input
          id="segmentation"
          type="checkbox"
          checked={toggles.showSegmentation}
          onChange={() => toggleLayer("showSegmentation")}
        />
        <label htmlFor="segmentation">Current perimeter</label>
      </div>
      <div className="flex items-center gap-2 text-sm">
        <input id="prediction" type="checkbox" checked={toggles.showPrediction} onChange={() => toggleLayer("showPrediction")} />
        <label htmlFor="prediction">Nowcast perimeters</label>
      </div>
      <div className="space-y-1 text-sm">
        <div className="flex items-center gap-2">
          <input
            id="uncertainty"
            type="checkbox"
            checked={toggles.showUncertainty}
            onChange={() => toggleLayer("showUncertainty")}
          />
          <label htmlFor="uncertainty">Uncertainty heatmap</label>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span>Opacity</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={toggles.uncertaintyOpacity}
            onChange={(event) => setUncertaintyOpacity(Number(event.target.value))}
            disabled={!toggles.showUncertainty}
          />
        </div>
      </div>
      <div className="flex items-center gap-2 text-sm">
        <input id="explain" type="checkbox" checked={toggles.showExplain} onChange={() => toggleLayer("showExplain")} />
        <label htmlFor="explain">Explain overlay</label>
      </div>
    </div>
  );
}

