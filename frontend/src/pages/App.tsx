import { EventPanel } from "../components/EventPanel";
import { LayerToggles } from "../components/LayerToggles";
import { MapView } from "../components/MapView";
import { TimeScrubber } from "../components/TimeScrubber";
import { DetailsDrawer } from "../components/DetailsDrawer";
import { useAppStore } from "../store/useAppStore";

export function App() {
  const { prediction } = useAppStore((state) => ({ prediction: state.prediction }));
  return (
    <div className="flex h-screen w-full flex-col md:flex-row bg-slate-50">
      <aside className="flex w-full flex-col border-b border-slate-200 bg-white md:h-full md:w-80 md:border-b-0 md:border-r">
        <div className="flex-1 overflow-y-auto"><EventPanel /></div>
        <div className="space-y-6 border-t border-slate-200 p-4">
          <TimeScrubber />
          <LayerToggles />
        </div>
      </aside>
      <main className="flex flex-1 flex-col md:flex-row">
        <section className="relative flex-1">
          <MapView />
          {!prediction && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-slate-400">
              Select an event to load predictions
            </div>
          )}
        </section>
        <aside className="w-full border-t border-slate-200 bg-white md:h-full md:w-72 md:border-t-0 md:border-l">
          <DetailsDrawer />
        </aside>
      </main>
    </div>
  );
}
