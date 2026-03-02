import { useMemories } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'

const typeColors: Record<string, string> = {
  decision: 'bg-violet-500/20 text-violet-400',
  pattern: 'bg-cyan-500/20 text-cyan-400',
  preference: 'bg-amber-500/20 text-amber-400',
  style: 'bg-pink-500/20 text-pink-400',
  habit: 'bg-emerald-500/20 text-emerald-400',
  insight: 'bg-blue-500/20 text-blue-400',
  context: 'bg-zinc-500/20 text-zinc-400',
}

export function RecentActivity() {
  const namespace = useNamespaceStore((s) => s.active)
  const { data, isLoading } = useMemories({
    namespace: namespace ?? undefined,
    limit: 15,
  })

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-850">
      <div className="border-b border-zinc-800 px-5 py-3">
        <h3 className="text-sm font-medium text-zinc-300">Recent Activity</h3>
      </div>
      <div className="divide-y divide-zinc-800/50">
        {isLoading ? (
          Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="flex items-start gap-3 px-5 py-3">
              <span className="mt-1 h-5 w-14 animate-pulse rounded bg-zinc-800" />
              <span className="h-4 w-full animate-pulse rounded bg-zinc-800" />
            </div>
          ))
        ) : !data?.memories?.length ? (
          <p className="px-5 py-8 text-center text-sm text-zinc-500">No memories yet</p>
        ) : (
          data.memories.map((m) => (
            <div key={m.id} className="flex items-start gap-3 px-5 py-3">
              <span
                className={`mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-xs font-medium ${
                  typeColors[m.memory_type] ?? typeColors.context
                }`}
              >
                {m.memory_type}
              </span>
              <p className="min-w-0 line-clamp-2 break-words text-sm text-zinc-300">{m.content}</p>
              <time className="ml-auto shrink-0 text-xs text-zinc-600">
                {new Date(m.created_at).toLocaleDateString()}
              </time>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
