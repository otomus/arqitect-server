import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import Layout from './Layout.vue'
import ArchDiagram from './ArchDiagram.vue'
import DreamDiagram from './DreamDiagram.vue'
import DataFlowDiagram from './DataFlowDiagram.vue'
import ToolIsolationDiagram from './ToolIsolationDiagram.vue'
import MemoryTiersDiagram from './MemoryTiersDiagram.vue'
import './arqitect.css'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('ArchDiagram', ArchDiagram)
    app.component('DreamDiagram', DreamDiagram)
    app.component('DataFlowDiagram', DataFlowDiagram)
    app.component('ToolIsolationDiagram', ToolIsolationDiagram)
    app.component('MemoryTiersDiagram', MemoryTiersDiagram)
  },
} satisfies Theme
