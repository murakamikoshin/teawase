import asyncio,json
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(args=['--autoplay-policy=no-user-gesture-required']); pg=await b.new_page(viewport={'width':390,'height':844}); errs=[]
        pg.on('pageerror',lambda e:errs.append(str(e)))
        await pg.add_init_script("(()=>{const AC=window.AudioContext;window.__n=0;const oc=AC.prototype.createOscillator;AC.prototype.createOscillator=function(){window.__n++;return oc.call(this)};window.__pl=[];const op=HTMLMediaElement.prototype.play;HTMLMediaElement.prototype.play=function(){if(this.tagName==='AUDIO')window.__pl.push(this.src.split('/').pop()+(this.loop?':loop':''));return op.call(this)};})()")
        await pg.goto('file:///home/claude/teawase/index.html')
        own={i:2 for i in list(range(1,21))+[31,101,84,68,404]}
        await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps({'owned':own,'enc':40,'joined':30,'mute':False,'tut':True,'tut2':True,'st0':1,'tgDone':True,'login':{'d':99991231,'n':1},'area':'forest','kira':{},'fav':84,'party':[84,19,31,15,404],'stone':3000}))
        await pg.reload(); await pg.wait_for_timeout(500)
        await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(2500)
        n1=await pg.evaluate("window.__n")
        for t in ['dex','shop','home']:
            await pg.click(f'#tab-{t}'); await pg.wait_for_timeout(400)
        await pg.wait_for_timeout(1500)
        print('osc before/after tabs',n1,await pg.evaluate("window.__n"),'audio plays',await pg.evaluate("window.__pl"),errs)
        await b.close()
asyncio.run(main())
