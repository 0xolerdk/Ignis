import { create } from "zustand";

import type { FeatureCollection } from "geojson";

import type { EventSummary } from "../lib/types";

export interface EventDetail extends EventSummary {
  geometry: FeatureCollection | null;
  sources?: string[];
}

export interface PredictionLayers {
  event_id: string;
  time_iso: string;
  horizons: number[];
  segmentation: FeatureCollection | null;
  predictions: Record<number, FeatureCollection>;
  stats: Record<number, { meanProbability: number; variance: number }>;
}

interface Toggles {
  showSegmentation: boolean;
  showPrediction: boolean;
  showUncertainty: boolean;
  showExplain: boolean;
  uncertaintyOpacity: number;
}

interface AppState {
  events: EventSummary[];
  selectedEventId?: string;
  eventDetail?: EventDetail;
  prediction?: PredictionLayers;
  horizons: number[];
  activeHorizon?: number;
  explainImage?: string;
  loading: boolean;
  error?: string;
  toggles: Toggles;
  fetchEvents: () => Promise<void>;
  selectEvent: (eventId: string) => Promise<void>;
  setHorizon: (horizon: number) => void;
  toggleLayer: (key: keyof Toggles, value?: boolean) => void;
  setUncertaintyOpacity: (value: number) => void;
  fetchExplain: () => Promise<void>;
}

const apiBase = import.meta.env.VITE_API_BASE ?? "/api";

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers ?? {}),
    },
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as T;
}

export const useAppStore = create<AppState>((set, get) => ({
  events: [],
  horizons: [],
  loading: false,
  toggles: {
    showSegmentation: true,
    showPrediction: true,
    showUncertainty: false,
    showExplain: false,
    uncertaintyOpacity: 0.45,
  },
  fetchEvents: async () => {
    set({ loading: true, error: undefined });
    try {
      const data = await fetchJSON<{ events: EventSummary[] }>(`${apiBase}/events`);
      set({ events: data.events, loading: false });
    } catch (error) {
      set({ error: (error as Error).message, loading: false });
    }
  },
  selectEvent: async (eventId: string) => {
    set({ selectedEventId: eventId, loading: true, error: undefined, prediction: undefined, explainImage: undefined });
    try {
      const detail = await fetchJSON<EventDetail>(`${apiBase}/events/${eventId}`);
      set({ eventDetail: detail });

      const body = {
        event_id: eventId,
        time_iso: detail.last_update ?? new Date().toISOString(),
        horizons: [6, 12, 24],
        mc_samples: 5,
      };
      const prediction = await fetchJSON<{
        event_id: string;
        time_iso: string;
        horizons: number[];
        segmentation: { polygons: GeoJSON.FeatureCollection | null; mean_probability: number };
        predictions: Array<{ horizon: number; perimeter: GeoJSON.FeatureCollection; mean_probability: number; variance: number }>;
      }>(`${apiBase}/predict`, {
        method: "POST",
        body: JSON.stringify(body),
      });

      const predictionLayers: PredictionLayers = {
        event_id: prediction.event_id,
        time_iso: prediction.time_iso,
        horizons: prediction.horizons,
        segmentation: prediction.segmentation.polygons,
        predictions: prediction.predictions.reduce((acc, item) => {
          acc[item.horizon] = item.perimeter;
          return acc;
        }, {} as Record<number, GeoJSON.FeatureCollection>),
        stats: prediction.predictions.reduce((acc, item) => {
          acc[item.horizon] = {
            meanProbability: item.mean_probability,
            variance: item.variance,
          };
          return acc;
        }, {} as Record<number, { meanProbability: number; variance: number }>),
      };

      set({
        prediction: predictionLayers,
        horizons: prediction.horizons,
        activeHorizon: prediction.horizons[0],
        loading: false,
        explainImage: undefined,
      });
      if (get().toggles.showExplain) {
        get().fetchExplain().catch(() => null);
      }
    } catch (error) {
      set({ error: (error as Error).message, loading: false });
    }
  },
  setHorizon: (horizon) => {
    set({ activeHorizon: horizon });
  },
  toggleLayer: (key, value) => {
    set((state) => {
      const nextValue = typeof value === "boolean" ? value : !state.toggles[key];
      const toggles = { ...state.toggles, [key]: nextValue };
      return {
        toggles,
        explainImage: key === "showExplain" && !nextValue ? undefined : state.explainImage,
      };
    });
    if (key === "showExplain") {
      const state = get();
      if (state.toggles.showExplain) {
        state.fetchExplain().catch(() => null);
      }
    }
  },
  setUncertaintyOpacity: (value) => {
    set((state) => ({ toggles: { ...state.toggles, uncertaintyOpacity: value } }));
  },
  fetchExplain: async () => {
    const state = get();
    if (!state.selectedEventId || !state.eventDetail) return;
    try {
      const explainUrl = `${apiBase}/explain?event_id=${encodeURIComponent(state.selectedEventId)}&time_iso=${encodeURIComponent(
        state.eventDetail.last_update ?? new Date().toISOString(),
      )}`;
      const response = await fetch(explainUrl);
      if (!response.ok) throw new Error(await response.text());
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const previous = state.explainImage;
      set({ explainImage: objectUrl });
      if (previous) {
        URL.revokeObjectURL(previous);
      }
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },
}));
