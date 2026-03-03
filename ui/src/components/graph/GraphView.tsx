import { useState, lazy, Suspense } from 'react'
import { Network, List, Box } from 'lucide-react'
import { GraphExplorer } from './GraphExplorer'

const ExplorerView = lazy(() =>
  import('./ExplorerView').then((m) => ({ default: m.ExplorerView }))
)
const VectorSpaceView = lazy(() =>
  import('./VectorSpaceView').then((m) => ({ default: m.VectorSpaceView }))
)

type SubView = 'graph' | 'explorer' | 'vectors'

const tabs: { id: SubView; label: string; icon: typeof Network }[] = [
  { id: 'graph', label: 'Graph', icon: Network },
  { id: 'explorer', label: 'Explorer', icon: List },
  { id: 'vectors', label: 'Vector Space', icon: Box },
]

export function GraphView() {
  const [active, setActive] = useState<SubView>('graph')

  return (
    <div className="flex h-full flex-col">
      {/* Sub-tab bar */}
      <div className="flex shrink-0 items-center gap-1 border-b border-zinc-800 px-2 pb-2">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActive(id)}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors ${
              active === id
                ? 'bg-zinc-800 text-zinc-100'
                : 'text-zinc-500 hover:bg-zinc-800/50 hover:text-zinc-300'
            }`}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="min-h-0 flex-1">
        {active === 'graph' && <GraphExplorer />}
        {active === 'explorer' && (
          <Suspense fallback={<LoadingFallback />}>
            <ExplorerView />
          </Suspense>
        )}
        {active === 'vectors' && (
          <Suspense fallback={<LoadingFallback />}>
            <VectorSpaceView />
          </Suspense>
        )}
      </div>
    </div>
  )
}

function LoadingFallback() {
  return (
    <div className="flex h-full items-center justify-center">
      <span className="text-sm text-zinc-500">Loading...</span>
    </div>
  )
}
