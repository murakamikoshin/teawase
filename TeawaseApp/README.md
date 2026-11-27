# テアワセ！ iOSアプリの外枠

ゲーム本体は `Teawase/Resources/index.html` の1ファイル。アプリは それを全画面で表示し、
広告・課金・共有・振動だけを受け持つ。**まだ一度もビルドしていない**(作った環境に Xcode が無い)。

## 作り方
1. `brew install xcodegen`
2. このフォルダで `xcodegen generate` → `Teawase.xcodeproj` ができる
3. Xcode で開いて、署名のチームを選んで実機か TestFlight で確認

## 公開までに やること
- App Store Connect に 非消耗型の商品を登録: `teawase.premium` `teawase.starter` `teawase.aura` `teawase.hero` `teawase.void`、消耗型で `teawase.seal1` `teawase.seal3`
- AdMob: Swift Package を追加し、Info.plist に `GADApplicationIdentifier`、`Ads.swift` の広告ユニットIDを自分のものに
  - SDK が無いあいだ、リリースビルドでは広告のごほうびは出ない(デバッグビルドだけ出る)
- `WebGameView.swift` の 利用規約・プライバシーのURL を本物に
- アプリのアイコン、スクリーンショット、年齢区分、確率表示の確認
- ゲームを更新するときは `Resources/index.html` を入れかえるだけ

## みんなで 大ボス(CloudKit)を 動かす 手順
ゲーム側は できている。アプリ側は `Teawase/Raid.swift` と `Teawase/Teawase.entitlements`。**まだ 一度も ビルドして いない**。
1. developer.apple.com → Identifiers → `com.koshinstudio.teawase` で **iCloud** に チェック → CloudKit の コンテナ `iCloud.com.koshinstudio.teawase` を 作って 結びつける
   (Xcode の Signing & Capabilities で「+ Capability → iCloud → CloudKit」に チェックして コンテナを 選んでも 同じ)
2. `xcodegen generate` で 作り直す(`project.yml` に entitlements を 足して ある)
3. CloudKit Console(icloud.developer.apple.com)→ コンテナを 選ぶ → Schema → Record Types → **RaidHit** を 作る
   - Fields: `day`(Int(64))・`dmg`(Int(64))・`tries`(Int(64))
   - Indexes: `day` に **QUERYABLE**、`recordName` に **QUERYABLE**
   - Security Roles は そのまま(_icloud = 作成できる / _creator = 自分のを 書ける / _world = 読める)
4. 実機(iCloud に サインイン済み)で バトル → 2ページ目「みんなで 大ボス」→ いどむ → 「みんなの 合計に 足す」。Console の Data で RaidHit が 1つ できて いれば OK
5. 公開の 前に Console の **Deploy Schema Changes…** で 本番(Production)へ 反映。これを 忘れると 審査版・公開版で 動かない
- 数字を 変える 所: ゲーム側 `RAID_TRY / RAID_GOAL / RAID_ME / RAID_RW`(index.html)、アプリ側 `capPerHit / maxTries`(Raid.swift)
- 弱い 所: 合計は その日の 全員ぶんを 読んで 足す 作り。1日の 参加が 数千人を こえたら、集計を サーバー側に 移す 必要が ある

## いっしょに でかける(Game Center)を 動かす 手順
ゲーム側は ブラウザの テスト用の つなぎで 6手の 流れと 切断を 自動テスト済み。アプリ側は `Teawase/Multi.swift`。**まだ 一度も ビルドして いない**。
1. developer.apple.com → Identifiers → `com.koshinstudio.teawase` で **Game Center** に チェック(Xcode の「+ Capability → Game Center」でも 同じ)
2. App Store Connect → アプリ → **Game Center** を 有効に する(リーダーボードや 実績は 作らなくて いい)
3. `xcodegen generate` → ビルド → TestFlight
4. 実機 **2台**(別々の Apple ID で Game Center に サインイン)で バトル → 2ページ目 →「あいてを さがす」。片方が ともだちを よぶ か、2台とも おまかせで さがす
5. たしかめる こと: 2台に 同じ 相手が 出る / 交互に 手が 出せる / 体力の 数字が 2台で 同じ / 片方の アプリを 落としても 残りが 続けられる / 30秒 何も しないと その 手が とぶ
- 通信の 中身(ゲーム側が 決めている): `{t:"start",mid,hands,kira,area}` `{t:"hit",slot,id,dmg,pf}` `{t:"bye"}`
- 弱い 所: けずった 量は 相手の 申告を 信じる(1手 3000 まで)。片方が 抜けても 体力は 2人ぶんの まま。シミュレーターでは Game Center の マッチングは ほぼ 試せない
