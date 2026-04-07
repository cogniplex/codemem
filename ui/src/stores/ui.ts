import { create } from 'zustand'

type View = 'dashboard' | 'graph' | 'memories' | 'insights'

interface UiState {
  sidebarCollapsed: boolean
  toggleSidebar: () => void
  activeView: View
  setActiveView: (view: View) => void
  searchOpen: boolean
  setSearchOpen: (open: boolean) => void
}

export const useUiStore = create<UiState>((set) => ({
  sidebarCollapsed: false,
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
  activeView: 'dashboard',
  setActiveView: (view) => set({ activeView: view }),
  searchOpen: false,
  setSearchOpen: (open) => set({ searchOpen: open }),
}))
