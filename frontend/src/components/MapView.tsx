import { useEffect, useMemo, useState } from "react";
import { GeoJSON, ImageOverlay, MapContainer, TileLayer, useMap } from "react-leaflet";
import type { LatLngBoundsExpression } from "leaflet";

import { computeBounds } from "../lib/geo";
import { useAppStore } from "../store/useAppStore";

function FitBounds({ bounds }: { bounds?: LatLngBoundsExpression }) {
  const map = useMap();
  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, { padding: [20, 20] });
    }
  }, [bounds, map]);
  return null;
}

function createGradientDataUrl(color: string, opacity: number): string {
  const canvas = document.createElement("canvas");
  canvas.width = 2;
  canvas.height = 2;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  ctx.fillStyle = color;
  ctx.globalAlpha = opacity;
  ctx.fillRect(0, 0, 2, 2);
  return canvas.toDataURL("image/png");
}

export function MapView() {
  const {
    eventDetail,
    prediction,
    activeHorizon,
    toggles,
    explainImage,
  } = useAppStore((state) => ({
    eventDetail: state.eventDetail,
    prediction: state.prediction,
    activeHorizon: state.activeHorizon,
    toggles: state.toggles,
    explainImage: state.explainImage,
  }));

  const bounds = useMemo(() => {
    if (prediction?.segmentation) {
      return computeBounds(prediction.segmentation);
    }
    if (eventDetail?.geometry) {
      return computeBounds(eventDetail.geometry);
    }
    return undefined;
  }, [eventDetail?.geometry, prediction?.segmentation]);

  const activeGeoJSON = useMemo(() => {
    if (!prediction || activeHorizon == null) return undefined;
    return prediction.predictions[activeHorizon];
  }, [prediction, activeHorizon]);

  const uncertaintyOverlay = useMemo(() => {
    if (!prediction || activeHorizon == null) return undefined;
    const stats = prediction.stats[activeHorizon];
    if (!stats) return undefined;
    const intensity = Math.min(1, stats.variance * 10);
    const color = `rgba(255,0,0,${Math.max(intensity, 0.1)})`;
    return createGradientDataUrl(color, 0.4 + intensity / 2);
  }, [prediction, activeHorizon]);

  const [mapKey] = useState(() => Date.now());

  return (
    <MapContainer key={mapKey} className="h-full w-full" center={[37.5, -120]} zoom={6} scrollWheelZoom>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="&copy; OpenStreetMap contributors" />
      <FitBounds bounds={bounds} />
      {toggles.showSegmentation && prediction?.segmentation && (
        <GeoJSON data={prediction.segmentation} pathOptions={{ color: "#2563eb", weight: 2 }} />
      )}
      {toggles.showPrediction && activeGeoJSON && (
        <GeoJSON data={activeGeoJSON} pathOptions={{ color: "#f97316", weight: 2, dashArray: "6 4" }} />
      )}
      {toggles.showUncertainty && bounds && uncertaintyOverlay && (
        <ImageOverlay url={uncertaintyOverlay} bounds={bounds} opacity={toggles.uncertaintyOpacity} />
      )}
      {toggles.showExplain && bounds && explainImage && (
        <ImageOverlay url={explainImage} bounds={bounds} opacity={0.6} />
      )}
    </MapContainer>
  );
}

