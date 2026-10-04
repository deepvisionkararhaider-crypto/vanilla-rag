// worker.js — 01-vanilla-rag
// Vanilla Retrieval-Augmented Generation on Cloudflare Workers.
//   ingest (txt/md) -> sentence-aware chunking -> Workers AI embeddings
//   -> D1 vector store -> cosine similarity search -> Groq generation + citations
import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { chunkText, embed, cosine, chat, ensureSchema, uid } from './core.js';

const app = new Hono();
app.use('/api/*', cors());

// ── Seed sample documents so the demo works instantly ─────────────────────────
const SAMPLE_DOCS = [
  {
    filename: 'rag-overview.md',
    content: `Retrieval-Augmented Generation (RAG) is an AI architecture that combines an information retrieval system with a large language model. Instead of relying only on parametric knowledge baked into the model weights, RAG retrieves relevant passages from an external knowledge base and injects them into the prompt as context. This grounds the model's answer in real documents and dramatically reduces hallucination.
The RAG pipeline has two phases. The ingestion phase parses documents, splits them into chunks, computes a vector embedding for each chunk, and stores those vectors in an index. The query phase embeds the user's question, performs a similarity search to retrieve the top-k most relevant chunks, builds a context window, and asks the LLM to answer using only that context.
Chunking strategy matters. Chunks that are too small lose context; chunks that are too large dilute the embedding and waste tokens. A common default is 512 tokens with an overlap of 50-100 tokens so that sentences spanning a boundary are not lost. Sentence-aware splitting, which never cuts mid-sentence, usually produces cleaner embeddings than fixed-character splitting.
Embeddings map text into a high-dimensional vector space where semantically similar texts are close together. Cosine similarity is the standard distance metric. A bi-encoder produces these embeddings quickly, which is why retrieval over millions of vectors is feasible, but the same model that embeds the query is not always the best judge of fine-grained relevance - that is where rerankers come in.
Citations are essential for trust. A trustworthy RAG system returns not only an answer but the exact source chunks that supported it, so a human can verify the claim. Production systems track a document id, chunk index, and similarity score for every retrieved passage.`,
  },
  {
    filename: 'what-is-ai.txt',
    content: `Artificial Intelligence (AI) is the field of computer science concerned with building systems that perform tasks that normally require human intelligence. These tasks include understanding language, recognizing images, making decisions, and generating new content.
Machine Learning (ML) is a subset of AI in which systems learn patterns from data rather than being explicitly programmed. Supervised learning trains on labelled examples, unsupervised learning finds structure in unlabelled data, and reinforcement learning learns from reward signals.
Deep Learning is a subset of machine learning based on artificial neural networks with many layers. Deep learning powers modern computer vision, speech recognition, and large language models. Transformers, introduced in 2017, are the dominant neural architecture and rely on an attention mechanism that weighs the relevance of every token to every other token.
Large Language Models (LLMs) are transformer-based models trained on enormous text corpora to predict the next token. Examples include the GPT family, Llama, and Mistral. LLMs can answer questions, summarize, translate, and write code, but they can also hallucinate facts and have a fixed training knowledge cutoff, which is precisely the problem that retrieval-augmented generation addresses.`,
  },
  {
    filename: 'embeddings-and-vectors.md',
    content: `A vector embedding is a dense numerical representation of text, an image, or audio in a continuous vector space. Text embeddings typically have between 384 and 3072 dimensions. The key property is that the distance between two vectors reflects semantic similarity: the vectors for "car" and "automobile" are close, while "car" and "banana" are far apart.
Common sentence embedding models include all-MiniLM-L6-v2 (384 dimensions), BGE-small (384 dimensions), and larger BGE or E5 variants. Smaller models are fast and cheap and are a good default for retrieval where speed matters; larger models improve accuracy at higher cost.
A vector database stores embeddings and supports fast approximate nearest neighbour search. Options include FAISS, Chroma, Qdrant, Weaviate, Pinecone, pgvector, and Cloudflare Vectorize. Exact search, which compares the query against every vector, is accurate but scales linearly; approximate methods such as HNSW trade a little accuracy for a large speed gain at scale.
Cosine similarity, dot product, and Euclidean distance are the usual metrics. For normalized embeddings, cosine similarity and dot product give the same ranking. Hybrid retrieval combines dense vector search with sparse lexical search such as BM25 to capture both semantic meaning and exact keyword matches.`,
  },
];

async function seedIfEmpty(env) {
  const row = await env.DB.prepare('SELECT COUNT(*) AS n FROM documents').first();
  if (row && row.n > 0) return;
  for (const doc of SAMPLE_DOCS) {
    await ingestDocument(env, doc.filename, doc.content);
  }
}

async function ingestDocument(env, filename, content) {
  const docId = uid();
  const chunks = chunkText(content, 800, 120);
  if (chunks.length === 0) throw new Error('Document produced no chunks');
  const vectors = await embed(env, chunks);
  const now = Date.now();
  const stmts = [
    env.DB.prepare('INSERT INTO documents (id, filename, content, created_at) VALUES (?,?,?,?)').bind(docId, filename, content, now),
  ];
  chunks.forEach((c, i) => {
    stmts.push(
      env.DB.prepare('INSERT INTO chunks (id, doc_id, filename, chunk_index, content, embedding, created_at) VALUES (?,?,?,?,?,?,?)')
        .bind(uid(), docId, filename, i, c, JSON.stringify(vectors[i]), now)
    );
  });
  await env.DB.batch(stmts);
  return { docId, chunks: chunks.length };
}

async function retrieve(env, query, topK, threshold) {
  const qv = (await embed(env, [query]))[0];
  const rows = (await env.DB.prepare(
    'SELECT id, filename, chunk_index, content, embedding FROM chunks LIMIT 5000'
  ).all()).results || [];
  const scored = rows.map((r) => ({
    chunk_id: r.id,
    filename: r.filename,
    chunk_index: r.chunk_index,
    content: r.content,
    score: cosine(qv, JSON.parse(r.embedding)),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.filter((s) => s.score >= threshold).slice(0, topK);
}

// ── API ───────────────────────────────────────────────────────────────────────
app.get('/api/health', (c) => c.json({ status: 'healthy', service: 'vanilla-rag', ts: Date.now() }));

app.get('/api/stats', async (c) => {
  await ensureSchema(c.env);
  const d = await c.env.DB.prepare('SELECT COUNT(*) AS n FROM documents').first();
  const ch = await c.env.DB.prepare('SELECT COUNT(*) AS n FROM chunks').first();
  return c.json({ documents: d?.n || 0, chunks: ch?.n || 0, embedding_model: '@cf/baai/bge-small-en-v1.5', llm_model: c.env.LLM_MODEL || 'openai/gpt-oss-120b' });
});

app.get('/api/documents', async (c) => {
  await ensureSchema(c.env);
  try { await seedIfEmpty(c.env); } catch {}
  const rows = (await c.env.DB.prepare(
    'SELECT d.id, d.filename, d.created_at, COUNT(ch.id) AS chunks FROM documents d LEFT JOIN chunks ch ON ch.doc_id = d.id GROUP BY d.id ORDER BY d.created_at DESC'
  ).all()).results || [];
  return c.json({ documents: rows, count: rows.length });
});

app.post('/api/documents', async (c) => {
  await ensureSchema(c.env);
  let body;
  try { body = await c.req.json(); } catch { return c.json({ error: 'Invalid JSON body' }, 400); }
  const filename = (body.filename || 'untitled.txt').toString().slice(0, 200);
  const text = (body.text || '').toString();
  if (!text.trim()) return c.json({ error: 'Field "text" is required and must be non-empty' }, 400);
  if (text.length > 200000) return c.json({ error: 'Document too large (max 200k chars)' }, 413);
  try {
    const t0 = Date.now();
    const { docId, chunks } = await ingestDocument(c.env, filename, text);
    return c.json({ document_id: docId, filename, chunk_count: chunks, ingestion_time_ms: Date.now() - t0 }, 201);
  } catch (e) {
    return c.json({ error: 'Ingestion failed', detail: String(e.message || e) }, 500);
  }
});

app.delete('/api/documents/:id', async (c) => {
  await ensureSchema(c.env);
  const id = c.req.param('id');
  await c.env.DB.batch([
    c.env.DB.prepare('DELETE FROM chunks WHERE doc_id = ?').bind(id),
    c.env.DB.prepare('DELETE FROM documents WHERE id = ?').bind(id),
  ]);
  return c.json({ deleted: id });
});

app.post('/api/search', async (c) => {
  await ensureSchema(c.env);
  let body;
  try { body = await c.req.json(); } catch { return c.json({ error: 'Invalid JSON body' }, 400); }
  const query = (body.query || '').toString();
  if (!query.trim()) return c.json({ error: 'Field "query" is required' }, 400);
  try { await seedIfEmpty(c.env); } catch {}
  const topK = Math.min(Math.max(parseInt(body.top_k) || 5, 1), 20);
  const threshold = typeof body.similarity_threshold === 'number' ? body.similarity_threshold : 0.2;
  const t0 = Date.now();
  const results = await retrieve(c.env, query, topK, threshold);
  return c.json({ query, results, count: results.length, retrieval_time_ms: Date.now() - t0 });
});

app.post('/api/query', async (c) => {
  await ensureSchema(c.env);
  let body;
  try { body = await c.req.json(); } catch { return c.json({ error: 'Invalid JSON body' }, 400); }
  const query = (body.query || '').toString();
  if (!query.trim()) return c.json({ error: 'Field "query" is required' }, 400);
  try { await seedIfEmpty(c.env); } catch {}
  const topK = Math.min(Math.max(parseInt(body.top_k) || 4, 1), 12);
  const threshold = typeof body.similarity_threshold === 'number' ? body.similarity_threshold : 0.2;

  const t0 = Date.now();
  const chunks = await retrieve(c.env, query, topK, threshold);
  const tRetrieve = Date.now();
  if (chunks.length === 0) {
    return c.json({
      query, answer: "I don't have enough information in the provided documents to answer this question.",
      citations: [], model: 'none', retrieval_time_ms: tRetrieve - t0, generation_time_ms: 0, total_time_ms: Date.now() - t0,
    });
  }
  const context = chunks.map((ch, i) => `[Source ${i + 1}] ${ch.filename} (chunk ${ch.chunk_index})\n${ch.content}`).join('\n\n---\n\n');
  const answer = await chat(c.env, [
    { role: 'system', content: 'You are a precise RAG assistant. Answer ONLY from the provided context. If the context is insufficient, say so. Cite sources inline in the form [Source N]. Never invent facts.' },
    { role: 'user', content: `Context:\n${context}\n\nQuestion: ${query}\n\nAnswer with citations:` },
  ]);
  const tGen = Date.now();
  return c.json({
    query,
    answer,
    citations: chunks.map((ch, i) => ({ n: i + 1, chunk_id: ch.chunk_id, filename: ch.filename, chunk_index: ch.chunk_index, similarity_score: Number(ch.score.toFixed(4)), content: ch.content.slice(0, 400) })),
    model: c.env.LLM_MODEL || 'openai/gpt-oss-120b',
    retrieval_time_ms: tRetrieve - t0,
    generation_time_ms: tGen - tRetrieve,
    total_time_ms: tGen - t0,
  });
});

// ── UI ────────────────────────────────────────────────────────────────────────
app.get('/', (c) => c.html(UI));

const UI = `<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vanilla RAG — Cloudflare Workers</title>
<script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css">
<style>body{background:#0b1020} .glass{background:rgba(255,255,255,.04);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.08)}</style>
</head>
<body class="min-h-screen text-slate-100">
<header class="border-b border-white/10">
  <div class="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-fuchsia-500 grid place-items-center"><i class="fas fa-magnifying-glass-chart"></i></div>
      <div><h1 class="font-bold text-lg leading-tight">Vanilla RAG</h1><p class="text-xs text-slate-400">Retrieval-Augmented Generation on Cloudflare Workers</p></div>
    </div>
    <div id="stats" class="text-xs text-slate-400"></div>
  </div>
</header>
<main class="max-w-6xl mx-auto px-4 py-6 grid lg:grid-cols-2 gap-6">
  <section class="glass rounded-2xl p-5">
    <h2 class="font-semibold mb-3"><i class="fas fa-file-arrow-up mr-2 text-indigo-400"></i>1. Knowledge base</h2>
    <div class="flex gap-2 mb-2">
      <input id="fname" placeholder="filename.txt" class="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm"/>
    </div>
    <textarea id="dtext" rows="4" placeholder="Paste text or load a .txt/.md file…" class="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm mb-2"></textarea>
    <div class="flex flex-wrap gap-2 mb-4">
      <button onclick="ingest()" class="bg-indigo-600 hover:bg-indigo-500 rounded-lg px-4 py-2 text-sm font-medium"><i class="fas fa-plus mr-1"></i>Ingest</button>
      <label class="cursor-pointer bg-white/10 hover:bg-white/20 rounded-lg px-3 py-2 text-sm"><i class="fas fa-paperclip mr-1"></i>Upload .txt/.md<input type="file" accept=".txt,.md,.markdown" class="hidden" onchange="loadFile(event)"></label>
      <button onclick="seed()" class="bg-white/5 hover:bg-white/10 rounded-lg px-3 py-2 text-sm">Load samples</button>
    </div>
    <h3 class="text-xs uppercase tracking-wide text-slate-400 mb-2">Documents (<span id="dcount">0</span>)</h3>
    <ul id="docs" class="space-y-1 max-h-52 overflow-auto text-sm"></ul>
  </section>
  <section class="glass rounded-2xl p-5">
    <h2 class="font-semibold mb-3"><i class="fas fa-comments mr-2 text-fuchsia-400"></i>2. Ask a question</h2>
    <div class="flex gap-2 mb-3">
      <input id="q" placeholder="e.g. What problem does RAG solve?" class="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm" onkeydown="if(event.key==='Enter')ask()"/>
      <button onclick="ask()" class="bg-fuchsia-600 hover:bg-fuchsia-500 rounded-lg px-4 py-2 text-sm font-medium"><i class="fas fa-paper-plane"></i></button>
    </div>
    <div id="meta" class="text-xs text-slate-400 mb-2"></div>
    <div id="answer" class="text-sm leading-relaxed whitespace-pre-wrap min-h-[80px]"></div>
    <div id="cites" class="mt-4 space-y-2"></div>
  </section>
</main>
<p class="text-center text-xs text-slate-500 py-6">Embeddings: Workers AI bge-small · LLM: Groq · Store: Cloudflare D1 · <span id="health">checking…</span></p>
<script>
const api = (p,o)=>fetch(p,o).then(r=>r.json());
async function refresh(){
  try{
    const s = await api('/api/stats');
    document.getElementById('stats').innerHTML = '<i class="fas fa-database mr-1"></i>'+s.documents+' docs · '+s.chunks+' chunks';
    const d = await api('/api/documents');
    document.getElementById('dcount').textContent = d.count;
    document.getElementById('docs').innerHTML = d.documents.map(x=>'<li class="flex items-center justify-between bg-white/5 rounded-lg px-3 py-2"><span class="truncate"><i class="fas fa-file-lines mr-2 text-slate-400"></i>'+x.filename+' <span class="text-slate-500">('+x.chunks+' chunks)</span></span><button onclick="del(\\''+x.id+'\\')" class="text-slate-400 hover:text-rose-400"><i class="fas fa-trash"></i></button></li>').join('') || '<li class="text-slate-500 text-xs px-3 py-2">No documents yet.</li>';
  }catch(e){}
}
async function ingest(){
  const filename=document.getElementById('fname').value||('doc-'+Date.now()+'.txt');
  const text=document.getElementById('dtext').value;
  if(!text.trim()){alert('Enter some text first');return;}
  const r=await fetch('/api/documents',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename,text})});
  const j=await r.json(); if(j.error){alert(j.error);return;}
  document.getElementById('dtext').value='';document.getElementById('fname').value='';refresh();
}
function loadFile(e){const f=e.target.files[0];if(!f)return;const r=new FileReader();r.onload=()=>{document.getElementById('dtext').value=r.result;document.getElementById('fname').value=f.name;};r.readAsText(f);}
async function del(id){await fetch('/api/documents/'+id,{method:'DELETE'});refresh();}
async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  document.getElementById('answer').innerHTML='<i class="fas fa-spinner fa-spin text-fuchsia-400"></i>';
  document.getElementById('cites').innerHTML='';document.getElementById('meta').textContent='';
  try{
    const r=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q})});
    const j=await r.json(); if(j.error){document.getElementById('answer').textContent='Error: '+j.error;return;}
    document.getElementById('answer').textContent=j.answer;
    document.getElementById('meta').innerHTML='<i class="fas fa-bolt mr-1 text-amber-400"></i>'+j.model+' · retrieve '+j.retrieval_time_ms+'ms · generate '+j.generation_time_ms+'ms · total '+j.total_time_ms+'ms';
    document.getElementById('cites').innerHTML = (j.citations||[]).map(c=>'<details class="bg-white/5 rounded-lg px-3 py-2 text-sm"><summary class="cursor-pointer"><span class="text-indigo-300 font-medium">[Source '+c.n+'] '+c.filename+'</span> <span class="text-slate-500">· chunk '+c.chunk_index+' · sim '+c.similarity_score+'</span></summary><p class="mt-2 text-slate-300 text-xs">'+c.content.replace(/</g,'&lt;')+'…</p></details>').join('');
  }catch(e){document.getElementById('answer').textContent='Request failed: '+e;}
}
async function seed(){document.getElementById('answer').innerHTML='<span class="text-slate-400 text-xs">Seeding sample documents…</span>';await api('/api/documents');refresh();document.getElementById('answer').innerHTML='';}
api('/api/health').then(h=>{document.getElementById('health').innerHTML='<span class="text-emerald-400">API '+h.status+'</span>';});
refresh();
</script>
</body></html>`;

export default app;
