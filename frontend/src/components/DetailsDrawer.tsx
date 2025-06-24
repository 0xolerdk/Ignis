import { useMemo } from "react";

import { downloadFeatureCollection } from "../lib/geo";
import { useAppStore } from "../store/useAppStore";

export function DetailsDrawer() {
  const {
    eventDetail,
    prediction,
    activeHorizon,
    toggles,
    horizons,
  } = useAppStore((state) => ({
    eventDetail: state.eventDetail,
    prediction: state.prediction,
    activeHorizon: state.activeHorizon,
    toggles: state.toggles,
    horizons: state.horizons,
  }));

  const stats = useMemo(() => {
    if (!prediction || activeHorizon == null) return undefined;
    return prediction.stats[activeHorizon];
  }, [prediction, activeHorizon]);

  if (!eventDetail || !prediction) {
    return (
      <div className="p-4 text-sm text-slate-500">Select an event to view prediction details.</div>
    );
  }

  return (
    <div className="space-y-4 p-4">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">{eventDetail.title}</h2>
        <p className="text-xs text-slate-500">
          Updated: {eventDetail.last_update ? new Date(eventDetail.last_update).toLocaleString() : "Unknown"}
        </p>
      </div>
      <div className="space-y-2 text-sm text-slate-700">
        <div className="flex justify-between">
          <span>Active horizon</span>
          <span>{activeHorizon ?? "--"}h</span>
        </div>
        {stats && (
          <>
            <div className="flex justify-between">
              <span>Mean probability</span>
              <span>{stats.meanProbability.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span>Variance</span>
              <span>{stats.variance.toFixed(3)}</span>
            </div>
          </>
        )}
      </div>
      <div className="space-y-2">
        <button
          className="w-full rounded border border-slate-300 px-3 py-2 text-sm hover:border-blue-500"
          onClick={() => prediction.segmentation && downloadFeatureCollection(prediction.segmentation, `${prediction.event_id}-segmentation.geojson`)}
          disabled={!prediction.segmentation}
        >
          Download current perimeter
        </button>
        <button
          className="w-full rounded border border-slate-300 px-3 py-2 text-sm hover:border-blue-500"
          onClick={() => {
            if (!prediction || activeHorizon == null) return;
            const collection = prediction.predictions[activeHorizon];
            if (collection) {
              downloadFeatureCollection(collection, `${prediction.event_id}-${activeHorizon}h.geojson`);
            }
          }}
          disabled={activeHorizon == null}
        >
          Download {activeHorizon ?? "--"}h forecast
        </button>
      </div>
      <div className="rounded border border-slate-200 bg-slate-50 p-3 text-xs text-slate-500">
        <p className="font-medium text-slate-600">Layers</p>
        <ul className="list-inside list-disc space-y-1">
          <li>Segmentation: {toggles.showSegmentation ? "visible" : "hidden"}</li>
          <li>Nowcast horizons: {toggles.showPrediction ? horizons.join(", ") : "hidden"}</li>
          <li>Uncertainty heatmap: {toggles.showUncertainty ? "on" : "off"}</li>
          <li>Explain overlay: {toggles.showExplain ? "enabled" : "disabled"}</li>
        </ul>
      </div>
    </div>
  );
}

