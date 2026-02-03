import { useEffect } from 'react'

export function useAutoPlay(
  isPlaying: boolean,
  activeLayer: number,
  totalLayers: number,
  onLayerChange: (layer: number) => void,
  onPlayingChange: (playing: boolean) => void,
  intervalMs = 800,
) {
  useEffect(() => {
    if (!isPlaying) return

    const id = setInterval(() => {
      onLayerChange((activeLayer + 1) % totalLayers)
      if (activeLayer + 1 >= totalLayers) {
        onPlayingChange(false)
      }
    }, intervalMs)

    return () => clearInterval(id)
  }, [isPlaying, activeLayer, totalLayers, onLayerChange, onPlayingChange, intervalMs])
}
