import asyncio,json
from playwright.async_api import async_playwright
async def play(pg):
    for turn in range(3):
        await pg.wait_for_selector('.pick:visible',timeout=15000); await pg.click('.pick:visible'); await pg.wait_for_timeout(500)
        for _ in range(40):
            hb=await pg.query_selector('#hit-btn:visible')
            if hb:
                box=await hb.bounding_box(); await pg.mouse.move(box['x']+box['width']/2,box['y']+box['height']/2); await pg.mouse.down(); await pg.wait_for_timeout(450); await pg.mouse.move(box['x']+box['width']/2+60,box['y']+box['height']/2-60); await pg.mouse.up()
            g=await pg.query_selector('.gauge[data-act=hit]:visible')
            if g: await g.click()
            await pg.wait_for_timeout(350)
            if await pg.query_selector('#anime:not([hidden])'): await pg.click('#anime')
            v=await pg.query_selector('video'); 
            if await pg.query_selector('.pick:visible') or await pg.query_selector('#raidexit-btn:visible'): break
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch()
        own={i:2 for i in list(range(1,21))+[31,101,84,68,404]}
        base={'owned':own,'enc':40,'joined':30,'mute':True,'tut':True,'st0':1,'tgDone':True,'login':{'d':99991231,'n':1},'area':'forest','kira':{},'fav':84,'party':[404,19,31,15,84],'stone':100}
        for mode in ('web','native'):
            pg=await b.new_page(viewport={'width':390,'height':844}); errs=[]; pg.on('pageerror',lambda e:errs.append(str(e)))
            if mode=='native':
                await pg.add_init_script("window.__msgs=[];window.webkit={messageHandlers:{teawase:{postMessage:function(m){window.__msgs.push(m);if(m.type==='raid'){var t=window.__tot=(window.__tot||12000)+(m.op==='hit'?m.dmg:0);window.__mine=(window.__mine||0)+(m.op==='hit'?m.dmg:0);window.__tr=(window.__tr||0)+(m.op==='hit'?1:0);setTimeout(function(){window.teawaseRaid({day:m.day,total:t,players:37,mine:window.__mine,tries:window.__tr,ok:true});},150);}}}}};")
            await pg.goto('file:///home/claude/teawase/index.html'); await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps(base)); await pg.reload(); await pg.wait_for_timeout(500)
            await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(400); await pg.click('#tab-arena'); await pg.wait_for_timeout(400)
            await pg.click('[data-act=pg][data-id="1"]:visible'); await pg.wait_for_timeout(500); await pg.screenshot(path=f'/tmp/rd_{mode}1.png')
            await pg.click('#raid-btn'); await pg.wait_for_timeout(600); await pg.screenshot(path=f'/tmp/rd_{mode}2.png')
            await play(pg); await pg.wait_for_timeout(800); await pg.screenshot(path=f'/tmp/rd_{mode}3.png')
            if mode=='native':
                await pg.click('#raidsend-btn'); await pg.wait_for_timeout(600)
            await pg.click('#raidexit-btn'); await pg.wait_for_timeout(700); await pg.screenshot(path=f'/tmp/rd_{mode}4.png')
            st=json.loads(await pg.evaluate("localStorage.getItem('teawase_v1')")); print(mode,st.get('raid'),st.get('stone'),st.get('shard'),st.get('enc'),errs)
            if mode=='native': print([m for m in await pg.evaluate("window.__msgs") if m['type']=='raid'])
            await pg.close()
        await b.close()
asyncio.run(main())
