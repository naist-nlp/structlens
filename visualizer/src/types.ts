export interface LayerData {
  layer: number;
  argmax_heads: number[];
}

export interface StructLensData {
  tokens: string[];
  layers: LayerData[];
}

export interface TreeNode {
  index: number;
  token: string;
  x: number;
  y: number;
  isRoot: boolean;
  isPadding: boolean;
}

export interface TreeEdge {
  source: number;
  target: number;
  sourceName: string;
  targetName: string;
}
