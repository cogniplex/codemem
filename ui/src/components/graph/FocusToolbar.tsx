import { ArrowLeft, Target } from 'lucide-react'

interface Props {
  nodeLabel: string
  depth: number
  onDepthChange: (depth: number) => void
  onExit: () => void
}

export function FocusToolbar({ nodeLabel, depth, onDepthChange, onExit }: Props) {
  return (
    <div className="absolute left-1/2 top-4 z-10 -translate-x-1/2">
      <div className="flex items-center gap-3 rounded-lg border border-violet-500/30 bg-zinc-900/95 px-4 py-2 shadow-lg backdrop-blur-sm">
        <Target size={14} className="text-violet-400" />
        <span className="max-w-48 truncate text-sm font-medium text-zinc-200">
          {nodeLabel}
        </span>

        <div className="flex items-center gap-1 rounded-md border border-zinc-700 bg-zinc-800 p-0.5">
          {[1, 2, 3].map((d) => (
            <button
              key={d}
              onClick={() => onDepthChange(d)}
              className={`rounded px-2 py-0.5 text-xs font-medium transition-colors ${
                depth === d
                  ? 'bg-violet-600 text-white'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              {d}
            </button>
          ))}
        </div>
        <span className="text-[10px] text-zinc-500">depth</span>

        <button
          onClick={onExit}
          className="flex items-center gap-1.5 rounded-md border border-zinc-700 bg-zinc-800 px-2.5 py-1 text-xs text-zinc-300 hover:border-zinc-600 hover:text-zinc-100"
        >
          <ArrowLeft size={12} />
          Full graph
        </button>
      </div>
    </div>
  )
}
