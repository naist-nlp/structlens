import { deflate, inflate } from 'pako';
import type { StructLensData } from './types';

interface CompactData {
  t: string[];
  l: number[][];
}

function toCompact(data: StructLensData): CompactData {
  return {
    t: data.tokens,
    l: data.layers.map((layer) => layer.argmax_heads),
  };
}

function fromCompact(compact: CompactData): StructLensData {
  return {
    tokens: compact.t,
    layers: compact.l.map((heads, i) => ({ layer: i, argmax_heads: heads })),
  };
}

function base64urlEncode(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function base64urlDecode(str: string): Uint8Array {
  const padded = str.replace(/-/g, '+').replace(/_/g, '/');
  const binary = atob(padded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

export function encodeTreeData(data: StructLensData): string {
  const compact = toCompact(data);
  const json = JSON.stringify(compact);
  const compressed = deflate(json);
  return base64urlEncode(compressed);
}

export function decodeTreeData(encoded: string): StructLensData | null {
  try {
    const bytes = base64urlDecode(encoded);
    const json = inflate(bytes, { to: 'string' });
    const compact = JSON.parse(json) as CompactData;
    if (!Array.isArray(compact.t) || !Array.isArray(compact.l)) {
      return null;
    }
    return fromCompact(compact);
  } catch {
    return null;
  }
}
