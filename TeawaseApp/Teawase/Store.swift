import StoreKit

/// 買い切り商品(非消耗型)。App Store Connect に 同じIDで登録する。
@MainActor
final class Store {
    static let shared = Store()
    /// ゲーム内のID → App Store の商品ID
    private let map: [String: String] = [
        "premium": "teawase.premium",
        "starter": "teawase.starter",
        "aura": "teawase.aura",
        "hero": "teawase.hero",
        "void": "teawase.void", "orrery": "teawase.orrery",
        "seal1": "teawase.seal1",   // 消耗型
        "seal3": "teawase.seal3",   // 消耗型
        "urtix": "teawase.urtix",   // 消耗型
        "welcome": "teawase.welcome",
        "stone1": "teawase.stone1", "stone2": "teawase.stone2", "stone3": "teawase.stone3", "gpick": "teawase.gpick",   // 消耗型
    ]

    func purchase(gameId: String) async -> Bool {
        guard let pid = map[gameId] else { return false }
        do {
            guard let product = try await Product.products(for: [pid]).first else { return false }
            let result = try await product.purchase()
            if case .success(let verification) = result, case .verified(let tx) = verification {
                await tx.finish()
                return true
            }
        } catch {}
        return false
    }

    /// 買ったことがある商品の ゲーム内ID
    func restore() async -> [String] {
        try? await AppStore.sync()
        var ids: [String] = []
        for await entitlement in Transaction.currentEntitlements {
            if case .verified(let tx) = entitlement, let game = map.first(where: { $0.value == tx.productID })?.key {
                ids.append(game)
            }
        }
        return ids
    }
}
