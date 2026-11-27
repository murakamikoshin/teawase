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
