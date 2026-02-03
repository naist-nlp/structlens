import { useRef, useEffect } from 'react';
import { initializeSvg, renderTree, resetZoom } from '../d3/render';
import { NODE_SPACING } from '../d3/layout';

interface TreeViewProps {
  tokens: string[];
  argmaxHeads: number[];
}

export function TreeView({ tokens, argmaxHeads }: TreeViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const prevHeadsRef = useRef<number[] | null>(null);
  const initializedRef = useRef(false);
  const prevTokensRef = useRef<string[] | null>(null);

  useEffect(() => {
    const svg = svgRef.current;
    const container = containerRef.current;
    if (!svg || !container) return;

    if (!initializedRef.current) {
      initializeSvg(svg);
      initializedRef.current = true;
    }

    // Reset zoom when tokens change (new file or range change)
    const tokensChanged = prevTokensRef.current === null
      || prevTokensRef.current.length !== tokens.length
      || prevTokensRef.current[0] !== tokens[0];

    if (tokensChanged) {
      const contentWidth = (tokens.length - 1) * NODE_SPACING + 2 * 50;
      const containerWidth = container.clientWidth;
      const offsetX = contentWidth < containerWidth
        ? (containerWidth - (tokens.length - 1) * NODE_SPACING) / 2
        : 50;
      // Compute baseline for reset
      let maxArcHeight = 40;
      for (let i = 0; i < argmaxHeads.length; i++) {
        const j = argmaxHeads[i];
        if (j === -1 || j === i) continue;
        const dist = Math.abs(i - j) * NODE_SPACING;
        const h = Math.min(dist * 0.5, 150);
        if (h > maxArcHeight) maxArcHeight = h;
      }
      const contentHeight = maxArcHeight + 30;
      const containerHeight = container.clientHeight;
      const baselineY = (containerHeight - contentHeight) / 2 + maxArcHeight;
      resetZoom(svg, offsetX, baselineY);
    }

    prevTokensRef.current = tokens;
    renderTree(svg, container, tokens, argmaxHeads, prevHeadsRef.current);
    prevHeadsRef.current = argmaxHeads;
  }, [tokens, argmaxHeads]);

  // Re-render on container resize
  useEffect(() => {
    const svg = svgRef.current;
    const container = containerRef.current;
    if (!svg || !container) return;

    const observer = new ResizeObserver(() => {
      if (initializedRef.current) {
        renderTree(svg, container, tokens, argmaxHeads, null);
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [tokens, argmaxHeads]);

  return (
    <div className="tree-view" ref={containerRef}>
      <svg ref={svgRef} />
      <style>{`
        .tree-view {
          flex: 1;
          overflow: hidden;
        }
        .tree-view svg {
          display: block;
          cursor: grab;
        }
        .tree-view svg:active {
          cursor: grabbing;
        }
      `}</style>
    </div>
  );
}
