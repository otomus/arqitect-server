import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Arqitect',
  description: 'A self-evolving AI agent server — sense, act, grow.',
  base: '/arqitect-server/',
  markdown: {
    theme: { light: 'github-light', dark: 'github-light' },
  },
  head: [
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    ['link', { href: 'https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;900&family=Share+Tech+Mono&display=swap', rel: 'stylesheet' }],
  ],
  themeConfig: {
    logo: false,
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Architecture', link: '/architecture/overview' },
      { text: 'Community', link: 'https://otomus.github.io/arqitect-community/' },
      { text: 'Monitoring', link: '/monitoring' },
      { text: 'Dashboard', link: 'https://otomus.github.io/arqitect-dashboard/' },
    ],
    sidebar: {
      '/guide/': [
        {
          text: 'Introduction',
          items: [
            { text: 'What is Arqitect?', link: '/guide/what-is-arqitect' },
            { text: 'Getting Started', link: '/guide/getting-started' },
            { text: 'Configuration', link: '/guide/configuration' },
          ],
        },
        {
          text: 'Core Concepts',
          items: [
            { text: 'Brain', link: '/guide/brain' },
            { text: 'Nerves', link: '/guide/nerves' },
            { text: 'Senses', link: '/guide/senses' },
            { text: 'Tools (MCP)', link: '/guide/tools' },
            { text: 'Memory', link: '/guide/memory' },
          ],
        },
        {
          text: 'Connectors',
          items: [
            { text: 'Bridge (Dashboard)', link: '/guide/bridge' },
            { text: 'Telegram', link: '/guide/telegram' },
            { text: 'WhatsApp', link: '/guide/whatsapp' },
          ],
        },
        {
          text: 'Advanced',
          items: [
            { text: 'Dream State', link: '/guide/dream-state' },
            { text: 'Personality', link: '/guide/personality' },
            { text: 'Community', link: '/guide/community' },
          ],
        },
      ],
      '/architecture/': [
        {
          text: 'Architecture',
          items: [
            { text: 'Overview', link: '/architecture/overview' },
            { text: 'Data Flow', link: '/architecture/data-flow' },
            { text: 'Memory Tiers', link: '/architecture/memory-tiers' },
            { text: 'Tool Isolation', link: '/architecture/tool-isolation' },
          ],
        },
      ],
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/otomus/arqitect-server' },
      { icon: 'discord', link: 'https://discord.gg/AYjJgWuS' },
      { icon: 'slack', link: 'https://join.slack.com/t/arqitectworkspace/shared_invite/zt-3t6zqf442-yMtBciurcPd_J66gZ1l1wA' },
    ],
  },
})
