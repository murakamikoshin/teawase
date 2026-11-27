import asyncio,json
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(); errs=[]
        for w,h,n in [(390,844,'a'),(375,667,'b')]:
            pg=await b.new_page(viewport={'width':w,'height':h}); pg.on('pageerror',lambda e:errs.append(str(e)))
            await pg.add_init_script("window.webkit={messageHandlers:{teawase:{postMessage:function(m){if(m.type==='raid'){setTimeout(function(){window.teawaseRaid({day:m.day,total:4200,players:23,mine:0,tries:0,ok:true});},100);}}}}};")
            await pg.goto('file:///home/claude/teawase/index.html')
            own={i:2 for i in list(range(1,21))+[31,101,84,68,404]}
            await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps({'owned':own,'enc':40,'joined':30,'mute':True,'tut':True,'tut2':True,'st0':1,'tgDone':True,'login':{'d':99991231,'n':1},'area':'forest','kira':{},'fav':84,'party':[84,19,31,15,404],'stone':3000}))
            await pg.reload(); await pg.wait_for_timeout(500); await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(300)
            await pg.click('#tab-shop'); await pg.wait_for_timeout(300); await pg.evaluate("document.querySelector('.shopx').scrollTop=document.querySelector('.hs').offsetTop-60"); await pg.wait_for_timeout(300); await pg.screenshot(path=f'/tmp/w9_{n}.png')
            await pg.click('#tab-home'); await pg.wait_for_timeout(300); await pg.click('#multi-btn'); await pg.wait_for_timeout(600); await pg.screenshot(path=f'/tmp/w8_{n}.png')
            await pg.close()
        print(errs); await b.close()
asyncio.run(main())
