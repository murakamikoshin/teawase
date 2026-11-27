import asyncio,json,sys
from playwright.async_api import async_playwright
JS="""()=>{var o=[];document.querySelectorAll('body,#home,#enc,#card,#anime,.encin,.page,main,section').forEach(function(e){if(!e.getClientRects().length)return;var s=e.scrollHeight-e.clientHeight;if(s>2&&getComputedStyle(e).overflowY!=='visible'&&getComputedStyle(e).overflowY!=='hidden')o.push((e.id||e.className||e.tagName)+':+'+s);});var c=document.querySelector('#enc:not([hidden]) .ctl');if(c){var bt=c.getBoundingClientRect().bottom;if(bt>innerHeight+1)o.push('CTL-CUT:'+Math.round(bt-innerHeight));var st=document.querySelector('#enc .stage');o.push('stage='+Math.round(st.getBoundingClientRect().height));}var v=document.querySelector('#app .view');if(v&&v.scrollHeight>v.clientHeight+1)o.push('VIEW-CUT:'+(v.scrollHeight-v.clientHeight));document.querySelectorAll('#app .pg,#app .grid9 .ci').forEach(function(e){if(e.scrollHeight>e.clientHeight+2)o.push('CLIP:'+e.className+(e.scrollHeight-e.clientHeight));});var d=document.scrollingElement;if(d.scrollHeight-d.clientHeight>2)o.push('DOC:+'+(d.scrollHeight-d.clientHeight));return o.join(' ');}"""
async def hit(pg):
    el=await pg.wait_for_selector('#hit-btn'); bx=await el.bounding_box()
    x=bx['x']+bx['width']/2-40; y=bx['y']+bx['height']/2
    await pg.mouse.move(x,y); await pg.mouse.down(); await pg.wait_for_timeout(120); await pg.mouse.move(x+80,y,steps=3); await pg.mouse.up()
async def run(p,w,h):
    b=await p.chromium.launch(args=['--use-gl=swiftshader','--enable-unsafe-swiftshader'])
    pg=await b.new_page(viewport={'width':w,'height':h})
    await pg.goto('file:///home/claude/teawase/index.html')
    own={i:2 for i in list(range(1,21))+[31,101]}
    await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps({'owned':own,'enc':40,'joined':30,'mute':True,'login':{'d':99991231,'n':1},'area':'forest','pity':12,'kira':{},'fav':15,'party':[15,19,31],'bond':{}}))
    await pg.reload(); await pg.wait_for_timeout(500)
    await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(300)
    async def rep(n):
        print(w,h,n,'|',await pg.evaluate(JS)); await pg.screenshot(path='/tmp/s_%d_%s.png'%(h,n))
    await rep('home')
    tabs=await pg.query_selector_all('.tabs button')
    for i in range(1,len(tabs)):
        tabs=await pg.query_selector_all('.tabs button'); await tabs[i].click(); await pg.wait_for_timeout(250); await rep('tab%d'%i)
    tabs=await pg.query_selector_all('.tabs button'); await tabs[0].click(); await pg.wait_for_timeout(200)
    await pg.click('#go-btn')
    async def clear():
        for _ in range(40):
            if await pg.query_selector('#anime:not([hidden])'): await pg.click('#anime')
            if await pg.query_selector('.pick:visible') or await pg.query_selector('#invite-btn'): return
            await pg.wait_for_timeout(250)
    await clear(); await rep('pick')
    await pg.click('.pick'); await pg.wait_for_timeout(200); await rep('gauge')
    await hit(pg); await pg.wait_for_timeout(1900)
    for _ in range(9):
        await clear()
        if await pg.query_selector('#invite-btn'): break
        await pg.click('.pick'); await hit(pg); await pg.wait_for_timeout(1900)
    await pg.wait_for_selector('#invite-btn',timeout=9000); await rep('chance')
    await pg.evaluate("window.__r=Math.random;Math.random=function(){return 0.0001}")
    await pg.click('#invite-btn'); await pg.wait_for_timeout(3000); await pg.evaluate("Math.random=window.__r"); await rep('joined')
    await pg.click('#again-btn'); await pg.wait_for_timeout(500); await clear()
    for _ in range(9):
        await clear()
        if await pg.query_selector('#invite-btn'): break
        await pg.click('.pick'); await hit(pg); await pg.wait_for_timeout(1900)
    await pg.wait_for_selector('#invite-btn',timeout=9000)
    await pg.evaluate("window.__r=Math.random;Math.random=function(){return 0.9999}")
    await pg.click('#invite-btn'); await pg.wait_for_timeout(3000); await pg.evaluate("Math.random=window.__r"); await rep('refused')
    await b.close()
async def main():
    async with async_playwright() as p:
        for w,h in ((390,844),(375,667),(430,932)):
            try: await run(p,w,h)
            except Exception as e: print(w,h,'ERR',str(e)[:200])
asyncio.run(main())
