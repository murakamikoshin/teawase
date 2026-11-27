import Foundation
import CloudKit

/// みんなで 大ボス: iCloud(CloudKit)の 公開データベースに「きょう じぶんが けずった 量」を 1人 1日 1レコードで 置き、
/// その日の 全員ぶんを 足して 合計を 出す。自前の サーバーは 使わない。
/// レコード型 RaidHit { day: Int64(検索できる), dmg: Int64, tries: Int64 }。レコード名は "hit-<日付>-<iCloudの利用者ID>"。
final class Raid {
    static let shared = Raid()
    static let containerID = "iCloud.com.koshinstudio.teawase"

    private let container = CKContainer(identifier: Raid.containerID)
    private var db: CKDatabase { container.publicCloudDatabase }
    private var userName: String?
    private let capPerHit = 3000      // 1回で 送れる 上限(ふつうの 遊び方では とどかない 値)
    private let maxTries = 3          // 1日に いどめる 回数
    private var cache: (day: Int, at: Date, total: Int, players: Int)?

    /// ゲームと 同じ 作り方の 日付(端末の 時計。20261127 の 形)
    static func today() -> Int {
        let c = Calendar.current.dateComponents([.year, .month, .day], from: Date())
        return (c.year ?? 0) * 10000 + (c.month ?? 0) * 100 + (c.day ?? 0)
    }

    private func user() async throws -> String {
        if let u = userName { return u }
        let id = try await container.userRecordID()
        userName = id.recordName
        return id.recordName
    }

    private func myRecordID(day: Int, user: String) -> CKRecord.ID {
        CKRecord.ID(recordName: "hit-\(day)-\(user)")
    }

    /// じぶんの きょうの レコード(まだ 無ければ nil)
    private func myRecord(day: Int, user: String) async throws -> CKRecord? {
        do {
            return try await db.record(for: myRecordID(day: day, user: user))
        } catch let e as CKError where e.code == .unknownItem {
            return nil
        }
    }

    /// その日の 全員ぶんを 足す(1ページ 最大件数ずつ、20ページまで)。60秒は 前の 結果を 使う
    private func totals(day: Int, fresh: Bool) async throws -> (Int, Int) {
        if !fresh, let c = cache, c.day == day, Date().timeIntervalSince(c.at) < 60 { return (c.total, c.players) }
        let query = CKQuery(recordType: "RaidHit", predicate: NSPredicate(format: "day == %d", day))
        var total = 0
        var players = 0
        var page = try await db.records(matching: query, desiredKeys: ["dmg"], resultsLimit: CKQueryOperation.maximumResults)
        var pages = 0
        while true {
            for (_, result) in page.matchResults {
                if case .success(let rec) = result {
                    total += Int(rec["dmg"] as? Int64 ?? 0)
                    players += 1
                }
            }
            pages += 1
            guard let cursor = page.queryCursor, pages < 20 else { break }
            page = try await db.records(continuingMatchFrom: cursor, desiredKeys: ["dmg"], resultsLimit: CKQueryOperation.maximumResults)
        }
        cache = (day, Date(), total, players)
        return (total, players)
    }

    private func fail(_ day: Int, _ error: Error) -> [String: Any] {
        var why = "net"
        if let e = error as? CKError, e.code == .notAuthenticated { why = "icloud" }
        return ["day": day, "ok": false, "err": why]
    }

    /// いまの 合計と じぶんの 量を 読む
    func status() async -> [String: Any] {
        let day = Raid.today()
        do {
            let u = try await user()
            let mine = try await myRecord(day: day, user: u)
            let (total, players) = try await totals(day: day, fresh: false)
            return ["day": day, "ok": true, "total": total, "players": players,
                    "mine": Int(mine?["dmg"] as? Int64 ?? 0), "tries": Int(mine?["tries"] as? Int64 ?? 0)]
        } catch {
            return fail(day, error)
        }
    }

    /// けずった 量を 足す(1日 maxTries 回まで、1回 capPerHit まで)
    func hit(dmg: Int) async -> [String: Any] {
        let day = Raid.today()
        do {
            let u = try await user()
            let rec = try await myRecord(day: day, user: u) ?? CKRecord(recordType: "RaidHit", recordID: myRecordID(day: day, user: u))
            let tries = Int(rec["tries"] as? Int64 ?? 0)
            var mine = Int(rec["dmg"] as? Int64 ?? 0)
            if tries < maxTries {
                mine += max(0, min(capPerHit, dmg))
                rec["day"] = Int64(day)
                rec["dmg"] = Int64(mine)
                rec["tries"] = Int64(tries + 1)
                _ = try await db.save(rec)
            }
            let (total, players) = try await totals(day: day, fresh: true)
            return ["day": day, "ok": true, "total": total, "players": players,
                    "mine": mine, "tries": min(maxTries, tries + 1)]
        } catch {
            return fail(day, error)
        }
    }
}
