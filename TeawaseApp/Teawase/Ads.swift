import UIKit
#if canImport(GoogleMobileAds)
import GoogleMobileAds
#endif

/// ごほうび広告。AdMob の SDK(Swift Package: swift-package-manager-google-mobile-ads)を入れると本物になる。
/// SDK が無いあいだは、デバッグビルドだけ「見たことにする」。リリースビルドでは報酬を出さない。
final class Ads {
    static let shared = Ads()
    /// 公開前に 自分の広告ユニットIDへ差しかえる(下は Google のテスト用ID)
    private let rewardedUnit = "ca-app-pub-3940256099942544/1712485313"

    func showRewarded(_ done: @escaping (Bool) -> Void) {
        #if canImport(GoogleMobileAds)
        GADRewardedAd.load(withAdUnitID: rewardedUnit, request: GADRequest()) { ad, error in
            guard let ad = ad, error == nil,
                  let root = UIApplication.shared.connectedScenes
                      .compactMap({ ($0 as? UIWindowScene)?.keyWindow }).first?.rootViewController
            else { done(false); return }
            var earned = false
            let delegate = Dismiss { done(earned) }
            ad.fullScreenContentDelegate = delegate
            objc_setAssociatedObject(ad, "dismiss", delegate, .OBJC_ASSOCIATION_RETAIN)
            ad.present(fromRootViewController: root) { earned = true }
        }
        #else
        #if DEBUG
        done(true)
        #else
        done(false)
        #endif
        #endif
    }
}

#if canImport(GoogleMobileAds)
private final class Dismiss: NSObject, GADFullScreenContentDelegate {
    let onClose: () -> Void
    init(_ onClose: @escaping () -> Void) { self.onClose = onClose }
    func adDidDismissFullScreenContent(_ ad: GADFullScreenPresentingAd) { onClose() }
    func ad(_ ad: GADFullScreenPresentingAd, didFailToPresentFullScreenContentWithError error: Error) { onClose() }
}
#endif
