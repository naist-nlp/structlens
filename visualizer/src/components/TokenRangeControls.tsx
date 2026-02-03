import { useCallback } from 'react';
import type { ChangeEvent } from 'react';

interface TokenRangeControlsProps {
  tokenStart: number;
  tokenEnd: number;
  totalTokens: number;
  onRangeChange: (start: number, end: number) => void;
}

export function TokenRangeControls({
  tokenStart,
  tokenEnd,
  totalTokens,
  onRangeChange,
}: TokenRangeControlsProps) {
  const handleStart = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const v = Math.max(0, Math.min(Number(e.target.value), tokenEnd - 1));
      onRangeChange(v, tokenEnd);
    },
    [tokenEnd, onRangeChange],
  );

  const handleEnd = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const v = Math.max(tokenStart + 1, Math.min(Number(e.target.value), totalTokens));
      onRangeChange(tokenStart, v);
    },
    [tokenStart, totalTokens, onRangeChange],
  );

  const handleReset = useCallback(() => {
    onRangeChange(0, totalTokens);
  }, [totalTokens, onRangeChange]);

  const isFullRange = tokenStart === 0 && tokenEnd === totalTokens;

  return (
    <div className="range-controls">
      <label className="range-field">
        <span className="range-field-label">From</span>
        <input
          type="number"
          min={0}
          max={tokenEnd - 1}
          value={tokenStart}
          onChange={handleStart}
          className="range-input"
        />
      </label>
      <label className="range-field">
        <span className="range-field-label">To</span>
        <input
          type="number"
          min={tokenStart + 1}
          max={totalTokens}
          value={tokenEnd}
          onChange={handleEnd}
          className="range-input"
        />
      </label>
      <span className="range-info">
        {tokenEnd - tokenStart} / {totalTokens}
      </span>
      {!isFullRange && (
        <button className="range-reset" onClick={handleReset}>
          Reset
        </button>
      )}

      <style>{`
        .range-controls {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .range-field {
          display: flex;
          align-items: center;
          gap: 4px;
        }
        .range-field-label {
          font-size: 16px;
          color: var(--text-muted);
        }
        .range-input {
          width: 60px;
          padding: 4px 6px;
          font-size: 16px;
          font-family: var(--font-mono);
          background: var(--bg);
          color: var(--text);
          border: 1px solid var(--border);
          border-radius: 4px;
          text-align: center;
        }
        .range-input:focus {
          outline: none;
          border-color: var(--accent);
        }
        .range-info {
          font-size: 16px;
          color: var(--text-muted);
          font-variant-numeric: tabular-nums;
        }
        .range-reset {
          font-size: 14px;
          padding: 3px 8px;
          border-radius: 4px;
          border: 1px solid var(--border);
          background: var(--bg);
          color: var(--text-muted);
          cursor: pointer;
          transition: border-color 0.15s;
        }
        .range-reset:hover {
          border-color: var(--accent);
          color: var(--text);
        }
      `}</style>
    </div>
  );
}
