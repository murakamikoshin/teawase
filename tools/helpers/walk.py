import asyncio,json,sys
from playwright.async_api import async_playwright
async def run(b,w,h,tag):
    pg=await b.new_page(viewport={'width':w,'height':h}); errs=[]; n=[0]
    pg.on('pageerror',lambda e:errs.append(str(e)))
    async def shot(name):
        n[0]+=1; await pg.screenshot(path=f'/tmp/w/{tag}_{n[0]:02d}_{name}.png')
    await pg.goto('file:///home/claude/teawase/index.html'); await pg.evaluate("localStorage.clear()"); await pg.evaluate("localStorage.setItem('teawase_v1',JSON.stringify({mute:true}))")
    await pg.reload(); await pg.wait_for_timeout(1500); await shot('title')
    await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(500)
    for i in range(6):
        if not await pg.query_selector('#tut-next'): break
        await shot(f'tut{i}'); await pg.click('#tut-next'); await pg.wait_for_timeout(350)
    await shot('starter'); await pg.click('#st0-1'); await pg.wait_for_timeout(200); await pg.click('#st0-ok'); await pg.wait_for_timeout(700)
    if await pg.evaluate("!document.getElementById('card').hidden"):
        await shot('card_after'); 
        if await pg.query_selector('#card-close'): await pg.click('#card-close')
    await pg.wait_for_timeout(300); await shot('after_starter')
    if await pg.query_selector('#tg-pull'):
        await pg.click('#tg-pull',force=True)
        for i in range(70):
            await pg.mouse.click(w/2,h/2); await pg.wait_for_timeout(350)
            if await pg.evaluate("!!document.querySelector('#tg-keep')"): break
        await pg.wait_for_timeout(600); await shot('tg_result'); await pg.click('#tg-keep')
        for i in range(40):
            await pg.mouse.click(w/2,h/3); await pg.wait_for_timeout(400)
            if await pg.evaluate("document.getElementById('gacha').hidden && !!document.querySelector('#go-btn')"): break
    await pg.wait_for_timeout(500); await shot('home')
    await pg.click('#go-btn'); await pg.wait_for_timeout(400); await shot('go')
    for _ in range(40):
        if await pg.query_selector('#anime:not([hidden])'): await pg.click('#anime')
        if await pg.query_selector('.pick:visible'): break
        await pg.wait_for_timeout(250)
    await shot('pick')
    for turn in range(9):
        if not await pg.query_selector('.pick:visible'): break
        await pg.click('.pick:visible',force=True); await pg.wait_for_timeout(500); await shot(f'atk{turn}')
        await pg.evaluate("window.__spFill&&0")
        for _ in range(30):
            hb=await pg.query_selector('#hit-btn:visible')
            if hb:
                box=await hb.bounding_box()
                await pg.mouse.move(box['x']+box['width']/2,box['y']+box['height']/2); await pg.mouse.down(); await pg.wait_for_timeout(450); await pg.mouse.move(box['x']+box['width']/2+60,box['y']+box['height']/2-60); await pg.mouse.up()
            g=await pg.query_selector('.gauge[data-act=hit]:visible')
            if g: await g.click()
            await pg.wait_for_timeout(350)
            if await pg.query_selector('#anime:not([hidden])'): await pg.click('#anime')
            if await pg.query_selector('.pick:visible') or await pg.query_selector('#invite-btn:visible'): break
    await pg.wait_for_timeout(600); await shot('chance')
    if await pg.query_selector('#invite-btn:visible'):
        await pg.click('#invite-btn',force=True)
        for _ in range(40):
            if await pg.query_selector('#anime:not([hidden])'): await pg.click('#anime')
            if await pg.query_selector('#home-btn:visible') or await pg.query_selector('#again-btn:visible') or await pg.query_selector('[data-act=home]:visible'): break
            await pg.wait_for_timeout(300)
        await pg.wait_for_timeout(800); await shot('result')
        hb=await pg.query_selector('[data-act=home]:visible')
        if hb: await hb.click()
    await pg.wait_for_timeout(800)
    if await pg.evaluate("!document.getElementById('card').hidden"):
        await shot('card2'); 
        if await pg.query_selector('#card-close'): await pg.click('#card-close')
    await pg.wait_for_timeout(300); await shot('home2')
    for t in ['gacha','dex','arena','shop']:
        await pg.click(f'#tab-{t}'); await pg.wait_for_timeout(450); await shot(t)
    await pg.click('#tab-arena'); await pg.wait_for_timeout(200)
    nx=await pg.query_selector('[data-act=pg][data-id="1"]:visible')
    if nx: await nx.click(); await pg.wait_for_timeout(300); await shot('arena2')
    await pg.click('#tab-guide'); await pg.wait_for_timeout(400); await shot('guide')
    for i in range(6):
        nx=await pg.query_selector('[data-act=pg][data-id="1"]:visible:not([disabled])')
        if not nx: break
        await nx.click(); await pg.wait_for_timeout(300); await shot(f'guide{i+2}')
    print(tag,n[0],errs); await pg.close()
async def main():
    import os; os.makedirs('/tmp/w',exist_ok=True)
    for f in os.listdir('/tmp/w'): os.remove('/tmp/w/'+f)
    async with async_playwright() as p:
        b=await p.chromium.launch()
        await run(b,390,844,'a'); await run(b,375,667,'b'); await b.close()
asyncio.run(main())
