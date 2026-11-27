import Foundation
import GameKit
import UIKit

/// いっしょに でかける(2人・同時): Game Center の マッチング(ともだちを よぶ / おまかせ)で 2人を つなぎ、
/// ゲームが 送って きた 文字列(JSON)を そのまま 相手へ わたす。中身の ルールは ゲーム側(index.html の MP)に ある。
/// ゲームへの 返事は onEvent で: ["ev": "matched", "host": Bool] / ["ev": "data", "data": String] / ["ev": "left"] / ["ev": "fail", "why": "gc" | "cancel" | "net"]
final class Multi: NSObject, GKMatchmakerViewControllerDelegate, GKMatchDelegate {
    static let shared = Multi()
    var onEvent: (([String: Any]) -> Void)?

    private var match: GKMatch?
    private var started = false

    private var rootVC: UIViewController? {
        UIApplication.shared.connectedScenes
            .compactMap { ($0 as? UIWindowScene)?.keyWindow }.first?.rootViewController
    }

    /// Game Center に サインイン(まだなら Apple の 画面が 出る)してから 相手さがしの 画面を 出す
    func find() {
        leave(notify: false)
        let me = GKLocalPlayer.local
        if me.isAuthenticated { showMatchmaker(); return }
        me.authenticateHandler = { [weak self] vc, error in
            guard let self = self else { return }
            if let vc = vc {
                self.rootVC?.present(vc, animated: true)
            } else if GKLocalPlayer.local.isAuthenticated {
                self.showMatchmaker()
            } else {
                self.onEvent?(["ev": "fail", "why": "gc"])
            }
        }
    }

    private func showMatchmaker() {
        let request = GKMatchRequest()
        request.minPlayers = 2
        request.maxPlayers = 2
        guard let vc = GKMatchmakerViewController(matchRequest: request) else {
            onEvent?(["ev": "fail", "why": "net"]); return
        }
        vc.matchmakerDelegate = self
        rootVC?.present(vc, animated: true)
    }

    func send(_ text: String) {
        guard let match = match, let data = text.data(using: .utf8) else { return }
        try? match.sendData(toAllPlayers: data, with: .reliable)
    }

    func leave(notify: Bool = false) {
        match?.delegate = nil
        match?.disconnect()
        match = nil
        started = false
        if notify { onEvent?(["ev": "left"]) }
    }

    /// 2人が そろったら 1回だけ 知らせる。先攻(ホスト)は ID の 文字列が 小さい ほう(2台で 同じ 答えに なる)
    private func startIfReady() {
        guard let match = match, !started, match.expectedPlayerCount == 0, let other = match.players.first else { return }
        started = true
        let host = GKLocalPlayer.local.gamePlayerID < other.gamePlayerID
        onEvent?(["ev": "matched", "host": host])
    }

    // MARK: GKMatchmakerViewControllerDelegate
    func matchmakerViewControllerWasCancelled(_ viewController: GKMatchmakerViewController) {
        viewController.dismiss(animated: true)
        onEvent?(["ev": "fail", "why": "cancel"])
    }

    func matchmakerViewController(_ viewController: GKMatchmakerViewController, didFailWithError error: Error) {
        viewController.dismiss(animated: true)
        onEvent?(["ev": "fail", "why": "net"])
    }

    func matchmakerViewController(_ viewController: GKMatchmakerViewController, didFind match: GKMatch) {
        viewController.dismiss(animated: true)
        self.match = match
        match.delegate = self
        startIfReady()
    }

    // MARK: GKMatchDelegate
    func match(_ match: GKMatch, player: GKPlayer, didChange state: GKPlayerConnectionState) {
        switch state {
        case .connected:
            startIfReady()
        case .disconnected:
            if started { leave(notify: true) }
        default:
            break
        }
    }

    func match(_ match: GKMatch, didReceive data: Data, fromRemotePlayer player: GKPlayer) {
        guard let text = String(data: data, encoding: .utf8) else { return }
        onEvent?(["ev": "data", "data": text])
    }

    func match(_ match: GKMatch, didFailWithError error: Error?) {
        if started { leave(notify: true) } else { onEvent?(["ev": "fail", "why": "net"]) }
    }
}
