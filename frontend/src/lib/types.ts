export interface EventSummary {
  event_id: string;
  title: string;
  last_update: string | null;
  area_sq_km: number | null;
}

export type EventDetail = EventSummary & {
  geometry: GeoJSON.FeatureCollection | null;
  sources?: string[];
};
