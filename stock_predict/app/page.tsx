import StockGraph from './StockGraph'

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <h1 className="text-4xl font-bold mb-8">Stock Prediction App</h1>
      <StockGraph />
    </main>
  )
}