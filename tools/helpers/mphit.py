import asyncio,json,sys
sys.argv=['x']
exec(open('/tmp/mp.py').read().split("async def run(")[0])
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(); ctx=await b.new_context(); errs=[]
        A=await openp(ctx,errs); B=await openp(ctx,errs)
        await A.click('#mp-find'); await A.wait_for_timeout(300); await B.click('#mp-find'); await B.wait_for_timeout(600)
        for _ in range(80):
            for pg in (A,B):
                if await pg.query_selector('#anime:not([hidden])'): await pg.click('#anime')
            if (await state(A))['pick']: break
            await A.wait_for_timeout(300)
        await move(A)
        for _ in range(40):
            if (await state(B))['pick']: break
            await B.wait_for_timeout(200)
        async def shots():
            for i in range(30):
                await A.screenshot(path=f'/tmp/mh_{i:02d}.png'); await A.wait_for_timeout(120)
        await asyncio.gather(move(B),shots())
        print(errs); await b.close()
asyncio.run(main())
