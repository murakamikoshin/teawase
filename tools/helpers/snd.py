import asyncio,json
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(args=['--autoplay-policy=no-user-gesture-required']); pg=await b.new_page(viewport={'width':390,'height':844}); errs=[]
        pg.on('pageerror',lambda e:errs.append(str(e))); pg.on('console',lambda m: errs.append(m.text) if m.type=='error' else None)
        await pg.add_init_script("""(()=>{const AC=window.AudioContext;window.__n=0;window.__nodes={};const oc=AC.prototype.createOscillator;AC.prototype.createOscillator=function(){window.__n++;return oc.call(this)};
          const og=AC.prototype.createDelay;AC.prototype.createDelay=function(x){window.__dl=(window.__dl||0)+1;window.__ac=this;return og.call(this,x)};})()""")
        await pg.goto('file:///home/claude/teawase/index.html')
        own={i:2 for i in list(range(1,21))+[31,101,84,68,404]}
        await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps({'owned':own,'enc':40,'joined':30,'mute':False,'tut':True,'st0':1,'tgDone':True,'login':{'d':99991231,'n':1},'area':'forest','kira':{},'fav':84,'party':[84,19,31,15,404],'stone':3000}))
        await pg.reload(); await pg.wait_for_timeout(500)
        await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(2500)
        n1=await pg.evaluate("window.__n")
        for t in ['dex','shop','arena','home']:
            await pg.click(f'#tab-{t}'); await pg.wait_for_timeout(300)
        await pg.click('#mute-btn'); await pg.wait_for_timeout(1500); n2=await pg.evaluate("window.__n"); await pg.wait_for_timeout(1500); n3=await pg.evaluate("window.__n")
        st=await pg.evaluate("localStorage.getItem('teawase_v1')"); print('bgmOff',json.loads(st).get('bgmOff'),json.loads(st).get('mute'))
        await pg.screenshot(path='/tmp/snd.png',clip={'x':0,'y':0,'width':390,'height':80})
        await pg.click('#mute-btn'); await pg.click('#mute-btn'); await pg.wait_for_timeout(300)
        await pg.click('#tab-gacha'); await pg.wait_for_timeout(300); await pg.click('#g-pull10')
        for i in range(60):
            await pg.mouse.click(195,420); await pg.wait_for_timeout(300)
            if await pg.evaluate("!!document.querySelector('.pe-cell')"): break
        print('osc',n1,n2,n3,await pg.evaluate("window.__n"),'delay',await pg.evaluate("window.__dl"),'state',await pg.evaluate("window.__ac&&window.__ac.state"))
        print(errs); await b.close()
asyncio.run(main())
