import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Autonomous Browser Agent | MAYINI Framework',
  description: 'Intelligent web automation powered by deep learning, vision transformers, and reinforcement learning.',
  keywords: ['browser agent', 'automation', 'AI', 'deep learning', 'web scraping', 'MAYINI'],
  authors: [{ name: 'Autonomous Browser Agent Team' }],
  openGraph: {
    title: 'Autonomous Browser Agent',
    description: 'Intelligent web automation powered by MAYINI Framework',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
      </head>
      <body className="font-sans antialiased">
        {children}
      </body>
    </html>
  )
}