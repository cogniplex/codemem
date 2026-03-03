import { Loader2, AlertTriangle } from 'lucide-react'
import { useVectors } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { PointCloud } from './PointCloud'

export function VectorSpaceView() {
  const namespace = useNamespaceStore((s) => s.active)
  const { data: points, isLoading, error } = useVectors(namespace ?? undefined)

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 size={24} className="animate-spin text-zinc-500" />
        <span className="ml-3 text-sm text-zinc-400">Loading embeddings...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center">
        <AlertTriangle size={20} className="text-amber-400" />
        <span className="ml-2 text-sm text-zinc-400">Failed to load vector data</span>
      </div>
    )
  }

  if (!points?.length) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-zinc-500">No embeddings found. Store memories first.</p>
      </div>
    )
  }

  return <PointCloud points={points} />
}
