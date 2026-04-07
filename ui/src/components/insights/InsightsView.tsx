import { useState } from 'react'
import { Lightbulb } from 'lucide-react'
import { ActivityTab } from './ActivityTab'
import { CodeHealthTab } from './CodeHealthTab'
import { SecurityTab } from './SecurityTab'
import { PerformanceTab } from './PerformanceTab'

const tabs = [
  { key: 'activity', label: 'Activity' },
  { key: 'code-health', label: 'Code Health' },
  { key: 'security', label: 'Security' },
  { key: 'performance', label: 'Performance' },
] as const

type TabKey = (typeof tabs)[number]['key']

const tabComponents: Record<TabKey, React.FC> = {
  activity: ActivityTab,
  'code-health': CodeHealthTab,
  security: SecurityTab,
  performance: PerformanceTab,
}

export function InsightsView() {
  const [activeTab, setActiveTab] = useState<TabKey>('activity')
  const TabContent = tabComponents[activeTab]

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div className="flex items-center gap-3">
        <div className="rounded-lg bg-violet-500/10 p-2">
          <Lightbulb size={18} className="text-violet-400" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-zinc-100">Insights</h2>
          <p className="text-[12px] text-zinc-500">
            Enriched analysis from git history, security scans, and performance profiling
          </p>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex items-center gap-0.5 rounded-xl border border-zinc-800/50 bg-zinc-900/50 p-1">
        {tabs.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`flex-1 rounded-lg px-4 py-2 text-[13px] font-medium transition-all ${
              activeTab === key
                ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <TabContent />
    </div>
  )
}
