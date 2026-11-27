import asyncio,json,sys
from playwright.async_api import async_playwright
own={i:2 for i in list(range(1,21))+[31,101,84,68,404]}
base={'owned':own,'enc':40,'joined':30,'mute':True,'tut':True,'st0':1,'tgDone':True,'login':{'d':99991231,'n':1},'area':'forest','kira':{},'fav':84,'party':[404,19,31,15,84],'stone':100}
async def openp(ctx,errs):
    pg=await ctx.new_page(); await pg.set_viewport_size({'width':390,'height':844}); pg.on('pageerror',lambda e:errs.append(str(e)))
    await pg.goto('file:///home/claude/teawase/index.html?mptest=1'); await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps(base)); await pg.reload(); await pg.wait_for_timeout(500)
    await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(400); await pg.click('#tab-arena'); await pg.wait_for_timeout(300)
    await pg.click('[data-act=pg][data-id="1"]:visible'); await pg.wait_for_timeout(400); return pg
async def state(pg):
    return await pg.evaluate("(()=>{const w=!!document.querySelector('#mp-left')&&!document.querySelector('.pick');return {pick:!!document.querySelector('.pick'),wait:w,inv:!!document.querySelector('#invite-btn'),hp:(document.querySelector('#hptxt')||{}).textContent,anime:!document.getElementById('anime').hidden}})()")
async def move(pg):
    await pg.click('.pick:visible'); await pg.wait_for_timeout(500)
    for _ in range(40):
        hb=await pg.query_selector('#hit-btn:visible')
        if hb:
            box=await hb.bounding_box(); await pg.mouse.move(box['x']+box['width']/2,box['y']+box['height']/2); await pg.mouse.down(); await pg.wait_for_timeout(450); await pg.mouse.move(box['x']+box['width']/2+60,box['y']+box['height']/2-60); await pg.mouse.up()
        g=await pg.query_selector('.gauge[data-act=hit]:visible')
        if g: await g.click()
        await pg.wait_for_timeout(350)
        if await pg.query_selector('#anime:not([hidden])'): await pg.click('#anime')
        st=await state(pg)
        if (st['wait'] or st['inv'] or st['pick']) and not await pg.query_selector('#hit-btn:visible') and not await pg.query_selector('.gauge[data-act=hit]:visible'): return
async def run(b,drop):
    ctx=await b.new_context(); errs=[]
    A=await openp(ctx,errs); B=await openp(ctx,errs)
    await A.click('#mp-find'); await A.wait_for_timeout(300); await B.click('#mp-find'); await B.wait_for_timeout(600)
    moves=0
    for step in range(200):
        for pg in (A,B):
            if pg.is_closed(): continue
            if await pg.query_selector('#anime:not([hidden])'): await pg.click('#anime')
        sa=await state(A); sb=None if B.is_closed() else await state(B)
        if step==0 or step%15==0: print(step,sa,sb)
        if sa['inv'] and (sb is None or sb['inv']): break
        if sa['pick'] and not sa['inv']:
            await move(A); moves+=1
            if drop and moves==1:
                await A.screenshot(path='/tmp/mp_a.png'); await B.screenshot(path='/tmp/mp_b.png')
        elif sb and sb['pick'] and not sb['inv']:
            await move(B); moves+=1
            if not drop and moves==2:
                await A.screenshot(path='/tmp/mp_a.png'); await B.screenshot(path='/tmp/mp_b.png')
        else: await A.wait_for_timeout(400)
        if drop and moves==2 and not B.is_closed():
            await B.evaluate("window.__mp.leave()"); await B.close(); print('B left')
    sa=await state(A); sb=None if B.is_closed() else await state(B)
    print('drop' if drop else 'full','moves',moves,'A',sa,'B',sb,errs)
    await A.screenshot(path='/tmp/mp_end_a.png')
    if not B.is_closed(): await B.screenshot(path='/tmp/mp_end_b.png')
    await ctx.close()
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(); await run(b,False); await run(b,True); await b.close()
asyncio.run(main())
