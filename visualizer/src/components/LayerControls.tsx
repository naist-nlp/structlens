import { useCallback } from 'react';
import type { ChangeEvent } from 'react';
import { useAutoPlay } from '../hooks/useAutoPlay';

interface LayerControlsProps {
  activeLayer: number;
  totalLayers: number;
  isPlaying: boolean;
  onLayerChange: (layer: number) => void;
  onPlayingChange: (playing: boolean) => void;
}

export function LayerControls({
  activeLayer,
  totalLayers,
  isPlaying,
  onLayerChange,
  onPlayingChange,
}: LayerControlsProps) {
  useAutoPlay(isPlaying, activeLayer, totalLayers, onLayerChange, onPlayingChange);

  const handleSlider = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onLayerChange(Number(e.target.value));
    },
    [onLayerChange],
  );

  const togglePlay = useCallback(() => {
    if (!isPlaying && activeLayer >= totalLayers - 1) {
      onLayerChange(0);
    }
    onPlayingChange(!isPlaying);
  }, [isPlaying, activeLayer, totalLayers, onLayerChange, onPlayingChange]);

  return (
    <div className="layer-controls">
      <button className="play-btn" onClick={togglePlay} title={isPlaying ? 'Pause' : 'Play'}>
        {isPlaying ? (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="4" width="4" height="16" />
            <rect x="14" y="4" width="4" height="16" />
          </svg>
        ) : (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <polygon points="5,3 19,12 5,21" />
          </svg>
        )}
      </button>
      <input
        type="range"
        min={0}
        max={totalLayers - 1}
        value={activeLayer}
        onChange={handleSlider}
        className="layer-slider"
      />
      <span className="layer-label">
        Layer {activeLayer} / {totalLayers - 1}
      </span>

      <style>{`
        .layer-controls {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .play-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 32px;
          height: 32px;
          border-radius: 6px;
          border: 1px solid var(--border);
          background: var(--bg);
          color: var(--text);
          cursor: pointer;
          padding: 0;
          transition: border-color 0.15s;
        }
        .play-btn:hover {
          border-color: var(--accent);
        }
        .layer-slider {
          width: 200px;
          accent-color: var(--accent);
          cursor: pointer;
        }
        .layer-label {
          font-size: 16px;
          color: var(--text-muted);
          font-variant-numeric: tabular-nums;
          min-width: 110px;
        }
      `}</style>
    </div>
  );
}
