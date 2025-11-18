import type { FeatureCollection, Geometry } from "geojson";
import type { LatLngBoundsExpression } from "leaflet";

export function computeBounds(collection?: FeatureCollection): LatLngBoundsExpression | undefined {
  if (!collection) return undefined;
  const coords: Array<[number, number]> = [];
  for (const feature of collection.features) {
    collectCoords(feature.geometry, coords);
  }
  if (coords.length === 0) return undefined;
  const lats = coords.map(([, lat]) => lat);
  const lngs = coords.map(([lng]) => lng);
  return [
    [Math.min(...lats), Math.min(...lngs)],
    [Math.max(...lats), Math.max(...lngs)],
  ];
}

function collectCoords(geometry: Geometry | null, target: Array<[number, number]>) {
  if (!geometry) return;
  if (geometry.type === "Point") {
    target.push(geometry.coordinates as [number, number]);
  } else if (geometry.type === "LineString" || geometry.type === "MultiPoint") {
    for (const coord of geometry.coordinates as Array<[number, number]>) {
      target.push(coord);
    }
  } else if (geometry.type === "Polygon") {
    for (const ring of geometry.coordinates) {
      for (const coord of ring as Array<[number, number]>) target.push(coord);
    }
  } else if (geometry.type === "MultiLineString" || geometry.type === "MultiPolygon") {
    for (const part of geometry.coordinates as Array<Array<[number, number]>>) {
      collectCoords({ type: geometry.type.includes("Polygon") ? "Polygon" : "LineString", coordinates: part } as Geometry, target);
    }
  } else if (geometry.type === "GeometryCollection") {
    for (const geom of geometry.geometries) {
      collectCoords(geom, target);
    }
  }
}

export function downloadFeatureCollection(collection: FeatureCollection, filename: string) {
  const blob = new Blob([JSON.stringify(collection, null, 2)], { type: "application/geo+json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}

