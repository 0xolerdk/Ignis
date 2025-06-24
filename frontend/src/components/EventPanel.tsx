import { useEffect } from "react";
import { useAppStore } from "../store/useAppStore";

export function EventPanel() {
  const {
    events,
    selectedEventId,
    fetchEvents,
    selectEvent,
    loading,
    error,
  } = useAppStore((state) => ({
    events: state.events,
    selectedEventId: state.selectedEventId,
    fetchEvents: state.fetchEvents,
    selectEvent: state.selectEvent,
    loading: state.loading,
    error: state.error,
  }));

  useEffect(() => {
    if (events.length === 0 && !loading) {
      fetchEvents().catch(() => null);
    }
  }, [events.length, loading, fetchEvents]);

  return (
    <div className="h-full overflow-y-auto p-4 space-y-3">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Wildfire Events</h1>
        <p className="text-xs text-slate-500">Select an event to view predictions and explanations.</p>
      </div>
      {error && <p className="text-sm text-red-500">{error}</p>}
      <ul className="space-y-2">
        {events.map((event) => (
          <li key={event.event_id}>
            <button
              type="button"
              onClick={() => selectEvent(event.event_id)}
              className={`w-full rounded border px-3 py-2 text-left text-sm transition ${
                selectedEventId === event.event_id
                  ? "border-blue-500 bg-blue-50 text-blue-900"
                  : "border-slate-200 hover:border-blue-400"
              }`}
            >
              <div className="font-medium">{event.title}</div>
              <div className="text-xs text-slate-600">
                Updated: {event.last_update ? new Date(event.last_update).toLocaleString() : "Unknown"}
              </div>
              {event.area_sq_km && (
                <div className="text-xs text-slate-500">Approx. area: {event.area_sq_km.toFixed(1)} km²</div>
              )}
            </button>
          </li>
        ))}
        {events.length === 0 && !loading && (
          <li className="text-xs text-slate-500">No events available.</li>
        )}
        {loading && <li className="text-xs text-slate-500">Loading...</li>}
      </ul>
    </div>
  );
}
