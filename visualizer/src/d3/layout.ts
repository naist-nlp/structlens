import type { TreeNode, TreeEdge } from '../types'

const NODE_SPACING = 90
const BASELINE_Y = 0

export function computeNodes(tokens: string[], argmaxHeads: number[]): TreeNode[] {
  return tokens.map((token, i) => ({
    index: i,
    token,
    x: i * NODE_SPACING,
    y: BASELINE_Y,
    isRoot: argmaxHeads[i] === i,
    isPadding: argmaxHeads[i] === -1,
  }))
}

export function computeEdges(tokens: string[], argmaxHeads: number[]): TreeEdge[] {
  const edges: TreeEdge[] = []
  for (let i = 0; i < argmaxHeads.length; i++) {
    const j = argmaxHeads[i]
    if (j === -1 || j === i) continue
    edges.push({
      source: i,
      target: j,
      sourceName: tokens[i],
      targetName: tokens[j],
    })
  }
  return edges
}

export function arcPath(x1: number, x2: number): string {
  const dist = Math.abs(x2 - x1)
  const height = Math.min(dist * 0.5, 150)
  return `M ${x1} 0 C ${x1} ${-height}, ${x2} ${-height}, ${x2} 0`
}

export { NODE_SPACING, BASELINE_Y }
