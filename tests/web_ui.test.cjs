// DOM unit tests only; no browser or GPU is started.
// NODE_PATH=/path/to/jsdom/node_modules node --test tests/web_ui.test.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {JSDOM} = require('jsdom');
const html = readFileSync('geocentric/web/index.html','utf8');
const script = readFileSync('geocentric/web/app.js','utf8');
const model = {name:'Test model',parameters:1000,context:512,device:'cpu',mode:'base',defaults:{mode:'auto',max_new_tokens:64}};
const tick = () => new Promise(resolve => setImmediate(resolve));
async function setup(t, saved = {}) {
  const dom = new JSDOM(html,{url:'http://localhost:8000',runScripts:'outside-only'});
  const w = dom.window;
  w.matchMedia = () => ({matches:false,addEventListener(){}});
  w.scrollTo = () => {};
  w.TextDecoder = TextDecoder;
  w.HTMLDialogElement.prototype.showModal = function(){this.open = true};
  w.HTMLDialogElement.prototype.close = function(){this.open = false};
  w.fetch = async () => ({ok:true,json:async()=>model});
  for (const [key,value] of Object.entries(saved)) w.localStorage.setItem(key,value);
  w.eval(script + '\nwindow.unit = {markdown,send,render,persistDraft};');
  await tick();
  t.after(() => w.close());
  return w;
}
test('mode-aware starter fills a draft without sending',async t=>{
  const w = await setup(t);
  assert.equal(w.document.querySelectorAll('.suggestion').length,3);
  assert.equal(w.document.querySelector('#mode-badge').textContent,'Text continuation');
  w.fetch = () => {throw Error('Suggestion must not send a request')};
  w.document.querySelector('.suggestion').click();
  assert.match(w.document.querySelector('#prompt').value,/forest/);
  assert.match(w.localStorage.getItem('geocentric.draft.new'),/forest/);
});
test('restores active conversation and its draft',async t=>{
  const chat={id:'a',title:'Saved chat',created:Date.now(),messages:[{role:'user',content:'Hello'}]};
  const w=await setup(t,{'geocentric.chats.v1':JSON.stringify([chat]),'geocentric.active.v1':JSON.stringify('a'),'geocentric.draft.a':'An unfinished thought'});
  assert.equal(w.document.querySelector('#prompt').value,'An unfinished thought');
  assert.match(w.document.title,/Saved chat/);
  assert.equal(w.document.querySelector('#conversation-title').textContent,'Saved chat');
  assert.equal(w.document.querySelector('[aria-current=page]').textContent,'Saved chat');
});
test('rename and undo deletion preserve messages',async t=>{
  const w=await setup(t,{'geocentric.chats.v1':JSON.stringify([{id:'a',title:'Before',created:Date.now(),messages:[{role:'user',content:'Find this content'}]}])});
  w.document.querySelector('.history-rename').click();
  w.document.querySelector('#chat-title').value='After';
  w.document.querySelector('#rename-form').dispatchEvent(new w.Event('submit',{cancelable:true}));
  assert.equal(w.document.querySelector('.history-title').textContent,'After');
  w.document.querySelector('#search').value='find this';
  w.document.querySelector('#search').dispatchEvent(new w.Event('input'));
  assert.equal(w.document.querySelectorAll('.history-item').length,1);
  w.document.querySelector('.history-delete').click();
  assert.equal(w.document.querySelectorAll('.history-item').length,0);
  w.document.querySelector('#toast button').click();
  const saved=JSON.parse(w.localStorage.getItem('geocentric.chats.v1'));
  assert.equal(saved[0].title,'After');
  assert.equal(saved[0].messages[0].content,'Find this content');
});
test('markdown supports structure without interpreting HTML or executable links',async t=>{
  const w=await setup(t), container=w.document.createElement('div');
  w.unit.markdown(container,'# Heading\n\n- First\n- Second\n\n> Quote\n\n[Docs](https://example.com)\n\n<script>alert(1)</script> [bad](javascript:alert(1))\n\n```js\n<img src=x onerror=alert(1)>\n```');
  assert.equal(container.querySelectorAll('li').length,2);
  assert.equal(container.querySelector('h2').textContent,'Heading');
  assert.equal(container.querySelector('blockquote').textContent,'Quote');
  assert.equal(container.querySelectorAll('a').length,1);
  assert.equal(container.querySelector('a').rel,'noopener noreferrer');
  assert.equal(container.querySelectorAll('script,img').length,0);
  assert.match(container.querySelector('pre').textContent,/<img/);
});
test('streaming preserves earlier message nodes',async t=>{
  const w=await setup(t);
  let continueStream, firstDelivered;
  const first=new Promise(r=>firstDelivered=r);
  const gate=new Promise(r=>continueStream=r);
  let n=0;
  w.fetch=async()=>({ok:true,body:{getReader:()=>({read:async()=>{
    n++;
    if(n===1)return {done:false,value:new TextEncoder().encode('{"type":"delta","text":"Hello"}\n')};
    if(n===2){firstDelivered();await gate;return {done:false,value:new TextEncoder().encode('{"type":"delta","text":" world"}\n')};}
    if(n===3)return {done:false,value:new TextEncoder().encode('{"type":"done","generated_tokens":2,"finish_reason":"stop"}\n')};
    return {done:true};
  }})}});
  w.document.querySelector('#prompt').value='Test';
  const done=w.unit.send();
  const user=w.document.querySelector('.message.user');
  await first;
  assert.equal(w.document.body.classList.contains("generating"),true);
  assert.equal(w.document.querySelector(".message.user").classList.contains("message-enter"),true);
  assert.ok(w.document.querySelector(".response-arriving"));
  assert.equal(w.document.querySelector('.message.user'),user);
  assert.equal(w.document.querySelector('.message.assistant .body').textContent,'Hello');
  continueStream();await done;
  assert.equal(w.document.querySelector('.message.assistant .body').textContent,'Hello world');
});
test('settings update mode and retain useful prompts',async t=>{
  const w=await setup(t);
  w.document.querySelector('#profile').click();
  w.document.querySelector('#mode').value='chat';
  w.document.querySelector('#save-settings').click();
  assert.equal(w.document.querySelector('#mode-badge').textContent,'Chat');
  assert.match(w.document.querySelector('#prompt').placeholder,/Ask/);
});

test('editing branches without changing original conversation',async t=>{
  const original={id:'root',title:'Original',messages:[{role:'user',content:'First question'},{role:'assistant',content:'First answer'},{role:'user',content:'Second question'},{role:'assistant',content:'Second answer'}]};
  const w=await setup(t,{'geocentric.chats.v1':JSON.stringify([original]),'geocentric.active.v1':JSON.stringify('root')});
  w.document.querySelector('[aria-label="Edit message"]').click();
  const chats=JSON.parse(w.localStorage.getItem('geocentric.chats.v1'));
  assert.deepEqual(chats.find(c=>c.id==='root'),original);
  assert.equal(chats[0].parentId,'root');
  assert.equal(chats[0].messages.length,0);
  assert.equal(w.document.querySelector('#prompt').value,'First question');
  assert.match(w.document.querySelector('#branch-banner').textContent,/Original/);
});
test('branch from an answer copies only its prefix and map navigates back',async t=>{
  const original={id:'root',title:'Original',messages:[{role:'user',content:'Question'},{role:'assistant',content:'Answer'},{role:'user',content:'Later'}]};
  const w=await setup(t,{'geocentric.chats.v1':JSON.stringify([original]),'geocentric.active.v1':JSON.stringify('root')});
  w.document.querySelector('[aria-label="Branch from here"]').click();
  const chats=JSON.parse(w.localStorage.getItem('geocentric.chats.v1'));
  assert.equal(chats[0].messages.length,2);
  assert.equal(chats.find(c=>c.id==='root').messages.length,3);
  w.document.querySelector('#conversation-map').click();
  assert.equal(w.document.querySelectorAll('.branch-node').length,2);
  w.document.querySelector('.branch-node').click();
  assert.equal(w.document.querySelectorAll('.message').length,3);
});
test('canvas uses a sandboxed named frame, edits code and unloads on close',async t=>{
  const w=await setup(t);let submitted=0;
  w.TextEncoder=TextEncoder;
  w.HTMLFormElement.prototype.submit=function(){submitted++};
  w.document.querySelector('#open-workshop').click();
  assert.equal(submitted,1);
  assert.match(w.document.querySelector('#canvas-payload').value,/<\/script>/);
  const frame=w.document.querySelector('#canvas-frame');
  assert.equal(frame.getAttribute('sandbox'),'allow-scripts');
  assert.equal(w.document.querySelector('#canvas-transport').target,frame.name);
  w.document.querySelector('#show-source').click();
  assert.equal(w.document.querySelector('#canvas-source').hidden,false);
  w.document.querySelector('#canvas-source').value='<h1>My artifact</h1>';
  w.document.querySelector('#run-canvas').click();
  assert.equal(w.document.querySelector('#canvas-payload').value,'<h1>My artifact</h1>');
  assert.equal(w.localStorage.getItem('geocentric.canvas.v1'),'<h1>My artifact</h1>');
  w.document.querySelector('#close-workshop').click();
  assert.equal(frame.src,'about:blank');
  assert.equal(w.document.querySelector('#workshop').hidden,true);
});
test('HTML code block previews only when explicitly opened',async t=>{
  const w=await setup(t);w.TextEncoder=TextEncoder;let submitted=0;
  w.HTMLFormElement.prototype.submit=()=>submitted++;
  const node=w.document.createElement('div');
  w.unit.markdown(node,'```html\n<h1>Preview me</h1>\n```');
  assert.equal(submitted,0);
  [...node.querySelectorAll('button')].find(b=>b.textContent==='Open in Canvas').click();
  assert.equal(submitted,1);
});

test('failed request restores draft without leaving a duplicate user turn',async t=>{
  const w=await setup(t);
  w.fetch=async()=>({ok:false,json:async()=>({error:'Model busy'})});
  w.document.querySelector('#prompt').value='Please retry this';
  await w.unit.send();
  assert.equal(w.document.querySelector('#prompt').value,'Please retry this');
  const chats=JSON.parse(w.localStorage.getItem('geocentric.chats.v1'));
  assert.equal(chats[0].messages.length,0);
  assert.match(w.document.querySelector('#error').textContent,/Model busy/);
});

test('conversation keyboard navigation preserves composer arrow keys',async t=>{
  const chat={id:'a',title:'Reading',messages:[{role:'user',content:'Hello'},{role:'assistant',content:'Hi'}]};
  const w=await setup(t,{'geocentric.chats.v1':JSON.stringify([chat]),'geocentric.active.v1':JSON.stringify('a')});
  const conversation=w.document.querySelector('#conversation'), rows=w.document.querySelectorAll('.message');
  conversation.focus();
  conversation.dispatchEvent(new w.KeyboardEvent('keydown',{key:'ArrowDown',bubbles:true}));
  assert.equal(w.document.activeElement,rows[0]);
  rows[0].dispatchEvent(new w.KeyboardEvent('keydown',{key:'ArrowDown',bubbles:true}));
  assert.equal(w.document.activeElement,rows[1]);
  const prompt=w.document.querySelector('#prompt');prompt.focus();
  prompt.dispatchEvent(new w.KeyboardEvent('keydown',{key:'ArrowUp',bubbles:true}));
  assert.equal(w.document.activeElement,prompt);
});
test('one settings entry and a noninteractive loaded-model label',async t=>{
  const w=await setup(t);
  assert.equal(w.document.querySelectorAll('button[aria-label="Settings"]').length,1);
  assert.equal(w.document.querySelector('#model-menu').tagName,'SPAN');
  w.document.querySelector('#profile').click();
  assert.equal(w.document.querySelector('#settings').open,true);
});
