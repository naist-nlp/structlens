import { useState, useCallback, useRef } from 'react'
import type { DragEvent, ChangeEvent } from 'react'
import type { StructLensData } from '../types'

interface FileDropZoneProps {
  onLoad: (data: StructLensData) => void
  onLoadSample?: () => void
}

function validate(json: unknown): json is StructLensData {
  if (typeof json !== 'object' || json === null) return false
  const obj = json as Record<string, unknown>
  if (!Array.isArray(obj.tokens)) return false
  if (!Array.isArray(obj.layers)) return false
  for (const layer of obj.layers) {
    if (typeof layer !== 'object' || layer === null) return false
    const l = layer as Record<string, unknown>
    if (typeof l.layer !== 'number') return false
    if (!Array.isArray(l.argmax_heads)) return false
  }
  return true
}

export function FileDropZone({ onLoad, onLoadSample }: FileDropZoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(
    (file: File) => {
      setError(null)
      if (!file.name.endsWith('.json')) {
        setError('Please drop a .json file')
        return
      }
      const reader = new FileReader()
      reader.onload = () => {
        try {
          const json: unknown = JSON.parse(reader.result as string)
          if (!validate(json)) {
            setError('Invalid format: expected { tokens: string[], layers: [{ layer, argmax_heads }] }')
            return
          }
          onLoad(json)
        } catch {
          setError('Failed to parse JSON')
        }
      }
      reader.readAsText(file)
    },
    [onLoad],
  )

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile],
  )

  const onDragOver = useCallback((e: DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const onDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const onInputChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFile(file)
    },
    [handleFile],
  )

  return (
    <div
      className={`dropzone ${isDragging ? 'dropzone--active' : ''}`}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".json"
        onChange={onInputChange}
        style={{ display: 'none' }}
      />
      <div className="dropzone-content">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        <p className="dropzone-label">
          Drop a StructLens JSON file here
        </p>
        <p className="dropzone-hint">or click to browse</p>
        {onLoadSample && (
          <button
            className="dropzone-sample-btn"
            onClick={(e) => { e.stopPropagation(); onLoadSample() }}
          >
            Try sample data
          </button>
        )}
        {error && <p className="dropzone-error">{error}</p>}
      </div>

      <div className="dropzone-format" onClick={(e) => e.stopPropagation()}>
        <p className="dropzone-format-title">Expected JSON format</p>
        <pre className="dropzone-format-code">{`{
  "tokens": ["Language", "exhibits", "inherent", ...],
  "layers": [
    {
      "layer": 0,
      "argmax_heads": [0, 0, 1, 0, 3, ...]
    },
    ...
  ]
}`}</pre>
        <ul className="dropzone-format-list">
          <li><code>tokens</code> — list of token strings</li>
          <li><code>layers[].layer</code> — layer index (number)</li>
          <li><code>layers[].argmax_heads</code> — dependency head for each token</li>
        </ul>
        <p className="dropzone-format-note">
          <code>argmax_heads[i] = j</code> → token i depends on j &nbsp;|&nbsp;
          <code>i = j</code> → root &nbsp;|&nbsp;
          <code>-1</code> → padding
        </p>
      </div>

      <style>{`
        .dropzone {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          border: 2px dashed var(--border);
          border-radius: 12px;
          margin: 24px;
          transition: border-color 0.2s, background 0.2s;
        }
        .dropzone:hover,
        .dropzone--active {
          border-color: var(--accent);
          background: var(--accent-dim);
        }
        .dropzone-content {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 12px;
          color: var(--text-muted);
        }
        .dropzone--active .dropzone-content {
          color: var(--accent);
        }
        .dropzone-label {
          font-size: 16px;
          font-weight: 500;
        }
        .dropzone-hint {
          font-size: 13px;
        }
        .dropzone-sample-btn {
          margin-top: 4px;
          padding: 6px 16px;
          font-size: 13px;
          font-weight: 500;
          border-radius: 6px;
          border: 1px solid var(--accent);
          background: var(--accent-dim);
          color: var(--accent);
          cursor: pointer;
          transition: background 0.15s, color 0.15s;
        }
        .dropzone-sample-btn:hover {
          background: var(--accent);
          color: #fff;
        }
        .dropzone-error {
          color: #f7768e;
          font-size: 13px;
          margin-top: 4px;
        }
        .dropzone-format {
          margin-top: 24px;
          padding: 16px 20px;
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: 8px;
          text-align: left;
          max-width: 480px;
          cursor: default;
        }
        .dropzone-format-title {
          font-size: 13px;
          font-weight: 600;
          color: var(--text);
          margin-bottom: 8px;
        }
        .dropzone-format-code {
          font-family: var(--font-mono);
          font-size: 12px;
          color: var(--text-muted);
          background: var(--bg);
          border: 1px solid var(--border);
          border-radius: 4px;
          padding: 10px 12px;
          margin: 0 0 10px;
          overflow-x: auto;
          white-space: pre;
        }
        .dropzone-format-list {
          list-style: none;
          padding: 0;
          margin: 0 0 8px;
          font-size: 12px;
          color: var(--text-muted);
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        .dropzone-format-list code {
          font-family: var(--font-mono);
          color: var(--text);
        }
        .dropzone-format-note {
          font-size: 11px;
          color: var(--text-muted);
          line-height: 1.6;
        }
        .dropzone-format-note code {
          font-family: var(--font-mono);
          color: var(--text);
        }
      `}</style>
    </div>
  )
}
