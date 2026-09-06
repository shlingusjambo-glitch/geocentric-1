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
  assert.equal(w.document.querySelector('.message.user'),user);
  assert.equal(w.document.querySelector('.message.assistant .body').textContent,'Hello');
  continueStream();await done;
  assert.equal(w.document.querySelector('.message.assistant .body').textContent,'Hello world');
});
test('settings update mode and retain useful prompts',async t=>{
  const w=await setup(t);
  w.document.querySelector('#settings-button').click();
  w.document.querySelector('#mode').value='chat';
  w.document.querySelector('#save-settings').click();
  assert.equal(w.document.querySelector('#mode-badge').textContent,'Chat');
  assert.match(w.document.querySelector('#prompt').placeholder,/Ask/);
});
