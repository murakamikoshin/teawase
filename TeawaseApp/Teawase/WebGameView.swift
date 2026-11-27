import SwiftUI
import WebKit
import StoreKit

/// ゲーム本体(Resources/index.html)を全画面で表示し、広告・課金・共有・振動だけをアプリ側で受け持つ。
/// ゲーム → アプリ: window.webkit.messageHandlers.teawase.postMessage({type: ...})
/// アプリ → ゲーム: teawaseNative(kind, id, ok)
struct WebGameView: UIViewRepresentable {
    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.mediaTypesRequiringUserActionForPlayback = []
        config.userContentController.add(context.coordinator, name: "teawase")
        let web = WKWebView(frame: .zero, configuration: config)
        web.isOpaque = false
        web.backgroundColor = .clear
        web.scrollView.isScrollEnabled = false
        web.scrollView.bounces = false
        web.scrollView.contentInsetAdjustmentBehavior = .never
        web.allowsLinkPreview = false
        context.coordinator.web = web
        if let url = Bundle.main.url(forResource: "index", withExtension: "html") {
            web.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
        }
        return web
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    final class Coordinator: NSObject, WKScriptMessageHandler {
        weak var web: WKWebView?
        // 公開前に 本物のURLへ差しかえる
        private let pages: [String: String] = [
            "terms": "https://example.com/teawase/terms",
            "privacy": "https://example.com/teawase/privacy",
        ]

        func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
            guard let body = message.body as? [String: Any], let type = body["type"] as? String else { return }
            switch type {
            case "ad":
                let token = body["token"] as? String ?? ""
                Ads.shared.showRewarded { [weak self] ok in self?.reply("ad", id: token, ok: ok) }
            case "buy":
                let id = body["id"] as? String ?? ""
                Task { @MainActor in
                    let ok = await Store.shared.purchase(gameId: id)
                    self.reply("buy", id: id, ok: ok)
                }
            case "restore":
                Task { @MainActor in
                    let ids = await Store.shared.restore()
                    let list = ids.map { "'\($0)'" }.joined(separator: ",")
                    self.web?.evaluateJavaScript("teawaseNative('restore',[\(list)],true)")
                }
            case "haptic":
                let strong = body["strong"] as? Bool ?? false
                UIImpactFeedbackGenerator(style: strong ? .heavy : .light).impactOccurred()
            case "share":
                share(dataURL: body["png"] as? String ?? "", name: body["name"] as? String ?? "")
            case "shareText":
                present([body["text"] as? String ?? ""])
            case "review":
                if let scene = UIApplication.shared.connectedScenes.first(where: { $0.activationState == .foregroundActive }) as? UIWindowScene {
                    SKStoreReviewController.requestReview(in: scene)
                }
            case "raid":
                // みんなで 大ボス: op "get" = いまの 合計を 読む / "hit" = けずった 量を 足す。結果は teawaseRaid({...}) で 返す
                let op = body["op"] as? String ?? "get"
                let dmg = body["dmg"] as? Int ?? 0
                Task { @MainActor in
                    let result = op == "hit" ? await Raid.shared.hit(dmg: dmg) : await Raid.shared.status()
                    self.sendJSON("teawaseRaid", result)
                }
            case "open":
                if let key = body["page"] as? String, let s = pages[key], let url = URL(string: s) {
                    UIApplication.shared.open(url)
                }
            default:
                break
            }
        }

        /// 辞書を JSON に して ゲーム側の 関数に 渡す
        private func sendJSON(_ fn: String, _ obj: [String: Any]) {
            guard let data = try? JSONSerialization.data(withJSONObject: obj),
                  let json = String(data: data, encoding: .utf8) else { return }
            DispatchQueue.main.async {
                self.web?.evaluateJavaScript("\(fn)(\(json))")
            }
        }

        private func reply(_ kind: String, id: String, ok: Bool) {
            let safe = id.replacingOccurrences(of: "'", with: "")
            DispatchQueue.main.async {
                self.web?.evaluateJavaScript("teawaseNative('\(kind)','\(safe)',\(ok ? "true" : "false"))")
            }
        }

        private func share(dataURL: String, name: String) {
            guard let comma = dataURL.firstIndex(of: ","),
                  let data = Data(base64Encoded: String(dataURL[dataURL.index(after: comma)...])),
                  let image = UIImage(data: data),
                  let root = UIApplication.shared.connectedScenes
                      .compactMap({ ($0 as? UIWindowScene)?.keyWindow }).first?.rootViewController
            else { return }
            _ = root
            present([image, "\(name) を なかまにした！ #テアワセ"])
        }

        private func present(_ items: [Any]) {
            guard let root = UIApplication.shared.connectedScenes
                .compactMap({ ($0 as? UIWindowScene)?.keyWindow }).first?.rootViewController else { return }
            let vc = UIActivityViewController(activityItems: items, applicationActivities: nil)
            vc.popoverPresentationController?.sourceView = root.view
            root.present(vc, animated: true)
        }
    }
}
