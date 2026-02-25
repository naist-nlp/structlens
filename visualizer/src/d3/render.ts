import * as d3 from 'd3'
import type { TreeNode, TreeEdge } from '../types'
import { computeNodes, computeEdges, arcPath, NODE_SPACING } from './layout'

const TRANSITION_MS = 400
const ARC_OPACITY = 0.6
const PADDING_X = 50
const TOKEN_BELOW = 30
const MIN_ARC_HEIGHT = 40

let zoomBehavior: d3.ZoomBehavior<SVGSVGElement, unknown> | null = null
let currentHoveredIndex: number | null = null

function applyHighlight(
  nodesGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
  arcsGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
  hoveredIndex: number | null,
  mode: 'children' | 'parents',
  edges: TreeEdge[],
) {
  const highlightedNodes = new Set<number>()
  const highlightedEdgeKeys = new Set<string>()

  if (hoveredIndex !== null) {
    highlightedNodes.add(hoveredIndex)
    if (mode === 'children') {
      // nodes that point TO the hovered node (its children)
      for (const edge of edges) {
        if (edge.target === hoveredIndex) {
          highlightedNodes.add(edge.source)
          highlightedEdgeKeys.add(`${edge.source}->${edge.target}`)
        }
      }
    } else {
      // node that hovered points TO (its parent)
      for (const edge of edges) {
        if (edge.source === hoveredIndex) {
          highlightedNodes.add(edge.target)
          highlightedEdgeKeys.add(`${edge.source}->${edge.target}`)
        }
      }
    }
  }

  const isHovering = hoveredIndex !== null

  nodesGroup.selectAll<SVGGElement, TreeNode>('g.node').each(function (d) {
    const isHighlighted = highlightedNodes.has(d.index)
    const isHovered = d.index === hoveredIndex
    const dimmed = isHovering && !isHighlighted

    d3.select(this)
      .select('circle')
      .attr('r', isHovered ? 6 : 4)
      .attr('opacity', d.isPadding ? 0 : dimmed ? 0.15 : 1)

    d3.select(this)
      .select('text')
      .attr('opacity', d.isPadding ? 0.3 : dimmed ? 0.15 : 1)
      .attr('font-weight', isHighlighted ? 'bold' : 'normal')
  })

  arcsGroup.selectAll<SVGPathElement, TreeEdge>('path.arc').each(function (d) {
    const key = `${d.source}->${d.target}`
    const isHighlightedEdge = highlightedEdgeKeys.has(key)
    const opacity = isHovering ? (isHighlightedEdge ? 0.9 : 0.05) : ARC_OPACITY
    const strokeWidth = isHighlightedEdge ? 2.5 : 1.5

    d3.select(this).attr('opacity', opacity).attr('stroke-width', strokeWidth)
  })
}

export function initializeSvg(svg: SVGSVGElement) {
  const sel = d3.select(svg)
  sel.selectAll('*').remove()

  const defs = sel.append('defs')
  defs
    .append('marker')
    .attr('id', 'arrowhead')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 8)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-4L10,0L0,4')
    .attr('fill', 'var(--arc-stroke)')

  const g = sel.append('g').attr('class', 'tree-group')
  g.append('g').attr('class', 'arcs-group')
  g.append('g').attr('class', 'nodes-group')

  // Set up zoom
  zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
    .scaleExtent([0.1, 5])
    .on('zoom', (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
      g.attr('transform', event.transform.toString())
    })

  sel.call(zoomBehavior)
}

export function resetZoom(svg: SVGSVGElement, offsetX: number, baselineY: number) {
  if (!zoomBehavior) return
  const sel = d3.select(svg)
  const transform = d3.zoomIdentity.translate(offsetX, baselineY)
  sel.call(zoomBehavior.transform, transform)
}

export function renderTree(
  svg: SVGSVGElement,
  container: HTMLElement,
  tokens: string[],
  argmaxHeads: number[],
  prevHeads: number[] | null,
  highlightMode: 'children' | 'parents' = 'children',
) {
  const nodes = computeNodes(tokens, argmaxHeads)
  const edges = computeEdges(tokens, argmaxHeads)

  const sel = d3.select(svg)
  const treeGroup = sel.select<SVGGElement>('.tree-group')
  const arcsGroup = treeGroup.select<SVGGElement>('.arcs-group')
  const nodesGroup = treeGroup.select<SVGGElement>('.nodes-group')

  // Compute max arc height from actual edges
  let maxArcHeight = MIN_ARC_HEIGHT
  for (const edge of edges) {
    const dist = Math.abs(edge.source - edge.target) * NODE_SPACING
    const h = Math.min(dist * 0.5, 150)
    if (h > maxArcHeight) maxArcHeight = h
  }

  const contentWidth = (tokens.length - 1) * NODE_SPACING + 2 * PADDING_X
  const contentHeight = maxArcHeight + TOKEN_BELOW
  const containerWidth = container.clientWidth
  const containerHeight = container.clientHeight

  // SVG fills the container; zoom handles overflow
  sel.attr('width', containerWidth).attr('height', containerHeight)

  const offsetX = contentWidth < containerWidth
    ? (containerWidth - (tokens.length - 1) * NODE_SPACING) / 2
    : PADDING_X

  const baselineY = (containerHeight - contentHeight) / 2 + maxArcHeight

  // Set the initial zoom transform (translate only, no scale) if no user zoom active
  if (zoomBehavior) {
    const currentTransform = d3.zoomTransform(svg)
    // Only set initial position if zoom is at identity (user hasn't zoomed/panned)
    if (currentTransform.k === 1 && currentTransform.x === 0 && currentTransform.y === 0) {
      const transform = d3.zoomIdentity.translate(offsetX, baselineY)
      sel.call(zoomBehavior.transform, transform)
    }
  } else {
    treeGroup.attr('transform', `translate(${offsetX}, ${baselineY})`)
  }

  // --- Arcs ---
  const prevEdgeMap = new Map<string, number[]>()
  if (prevHeads) {
    for (let i = 0; i < prevHeads.length; i++) {
      const j = prevHeads[i]
      if (j === -1 || j === i) continue
      prevEdgeMap.set(`${i}->${j}`, [i * NODE_SPACING, j * NODE_SPACING])
    }
  }

  arcsGroup
    .selectAll<SVGPathElement, TreeEdge>('path.arc')
    .data(edges, (d) => `${d.source}->${d.target}`)
    .join(
      (enter) => {
        const path = enter
          .append('path')
          .attr('class', 'arc')
          .attr('fill', 'none')
          .attr('stroke', 'var(--arc-stroke)')
          .attr('stroke-width', 1.5)
          .attr('marker-end', 'url(#arrowhead)')

        if (prevHeads) {
          path.each(function (d) {
            const prevKey = `${d.source}->${d.target}`
            const prev = prevEdgeMap.get(prevKey)
            if (prev) {
              d3.select(this).attr('d', arcPath(prev[0], prev[1]))
            } else {
              d3.select(this)
                .attr('d', arcPath(d.source * NODE_SPACING, d.target * NODE_SPACING))
                .attr('opacity', 0)
            }
          })

          path
            .transition()
            .duration(TRANSITION_MS)
            .attr('d', (d) => arcPath(d.source * NODE_SPACING, d.target * NODE_SPACING))
            .attr('opacity', ARC_OPACITY)
        } else {
          path
            .attr('d', (d) => arcPath(d.source * NODE_SPACING, d.target * NODE_SPACING))
            .attr('opacity', ARC_OPACITY)
        }

        return path
      },
      (update) => {
        update
          .transition()
          .duration(TRANSITION_MS)
          .attr('d', (d) => arcPath(d.source * NODE_SPACING, d.target * NODE_SPACING))
          .attr('opacity', ARC_OPACITY)
        return update
      },
      (exit) => {
        exit.transition().duration(TRANSITION_MS).attr('opacity', 0).remove()
        return exit
      },
    )

  // --- Nodes ---
  nodesGroup
    .selectAll<SVGGElement, TreeNode>('g.node')
    .data(nodes, (d) => d.index)
    .join(
      (enter) => {
        const g = enter
          .append('g')
          .attr('class', 'node')
          .attr('transform', (d) => `translate(${d.x}, ${d.y})`)

        g.append('circle')
          .attr('r', 4)
          .attr('cy', -2)
          .attr('fill', (d) =>
            d.isPadding ? 'transparent' : d.isRoot ? 'var(--root-color)' : 'var(--accent)',
          )
          .attr('opacity', (d) => (d.isPadding ? 0 : 1))

        g.append('text')
          .attr('text-anchor', 'middle')
          .attr('dy', 20)
          .attr('font-size', 14)
          .attr('font-family', 'var(--font-mono)')
          .attr('fill', (d) =>
            d.isPadding ? 'var(--text-muted)' : d.isRoot ? 'var(--root-color)' : 'var(--text)',
          )
          .attr('opacity', (d) => (d.isPadding ? 0.3 : 1))
          .text((d) => d.token)

        return g
      },
      (update) => {
        update
          .select('circle')
          .transition()
          .duration(TRANSITION_MS)
          .attr('fill', (d) =>
            d.isPadding ? 'transparent' : d.isRoot ? 'var(--root-color)' : 'var(--accent)',
          )
          .attr('opacity', (d) => (d.isPadding ? 0 : 1))

        update
          .select('text')
          .transition()
          .duration(TRANSITION_MS)
          .attr('fill', (d) =>
            d.isPadding ? 'var(--text-muted)' : d.isRoot ? 'var(--root-color)' : 'var(--text)',
          )
          .attr('opacity', (d) => (d.isPadding ? 0.3 : 1))

        return update
      },
      (exit) => exit.remove(),
    )

  // Attach hover handlers (re-attached on every render to keep highlightMode current)
  nodesGroup
    .selectAll<SVGGElement, TreeNode>('g.node')
    .on('mouseover', (_event, d) => {
      currentHoveredIndex = d.index
      applyHighlight(nodesGroup, arcsGroup, d.index, highlightMode, edges)
    })
    .on('mouseout', () => {
      currentHoveredIndex = null
      applyHighlight(nodesGroup, arcsGroup, null, highlightMode, edges)
    })

  // Re-apply current highlight state (e.g. mode toggled while hovering)
  applyHighlight(nodesGroup, arcsGroup, currentHoveredIndex, highlightMode, edges)
}
