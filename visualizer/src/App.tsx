import { useState, useCallback } from 'react';
import './App.css';
import type { StructLensData } from './types';
import { FileDropZone } from './components/FileDropZone';
import { LayerControls } from './components/LayerControls';
import { TokenRangeControls } from './components/TokenRangeControls';
import { TreeView } from './components/TreeView';
import sampleData from './sampleData.json';

const defaultData = sampleData as StructLensData;

function App() {
  const [data, setData] = useState<StructLensData | null>(defaultData);
  const [activeLayer, setActiveLayer] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [tokenStart, setTokenStart] = useState(0);
  const [tokenEnd, setTokenEnd] = useState(defaultData.tokens.length);
  const [showUpload, setShowUpload] = useState(false);

  const handleLoad = useCallback((d: StructLensData) => {
    setData(d);
    setTokenStart(0);
    setTokenEnd(d.tokens.length);
    setActiveLayer(0);
    setShowUpload(false);
  }, []);

  const totalLayers = data ? data.layers.length : 0;
  const totalTokens = data ? data.tokens.length : 0;

  // Slice tokens and remap argmax_heads to local indices
  let visibleTokens: string[] = [];
  let visibleHeads: number[] = [];
  if (data) {
    const fullHeads = data.layers[activeLayer].argmax_heads;
    visibleTokens = data.tokens.slice(tokenStart, tokenEnd);
    visibleHeads = fullHeads.slice(tokenStart, tokenEnd).map((h, localIdx) => {
      if (h === -1) return -1;
      if (h < tokenStart || h >= tokenEnd) return localIdx; // treat out-of-range targets as roots
      return h - tokenStart;
    });
  }

  return (
    <div className="app">
      <div className="app-header">
        <div className="app-title">
          <span>StructLens</span> Visualizer
        </div>
        {data && !showUpload && (
          <div className="app-controls">
            <button className="load-btn" onClick={() => setShowUpload(true)}>
              Upload File
            </button>
            <TokenRangeControls
              tokenStart={tokenStart}
              tokenEnd={tokenEnd}
              totalTokens={totalTokens}
              onRangeChange={(start, end) => { setTokenStart(start); setTokenEnd(end); }}
            />
            <LayerControls
              activeLayer={activeLayer}
              totalLayers={totalLayers}
              isPlaying={isPlaying}
              onLayerChange={setActiveLayer}
              onPlayingChange={setIsPlaying}
            />
          </div>
        )}
      </div>
      <div className="app-main">
        {data && !showUpload ? (
          <TreeView tokens={visibleTokens} argmaxHeads={visibleHeads} />
        ) : (
          <FileDropZone onLoad={handleLoad} onLoadSample={() => handleLoad(defaultData)} />
        )}
      </div>
    </div>
  );
}

export default App;
