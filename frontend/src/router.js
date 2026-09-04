import { createRouter, createWebHistory } from "vue-router";
import AppLayout from "./layouts/AppLayout.vue";
import Home from "./pages/Home.vue";
import Dashboard from "./pages/Dashboard.vue";
import FactorDetail from "./pages/FactorDetail.vue";
import Families from "./pages/Families.vue";
import FamilyEval from "./pages/FamilyEval.vue";
import FactorFilter from "./pages/FactorFilter.vue";
import Docs from "./pages/Docs.vue";
import BatchEval from "./pages/BatchEval.vue";
import InteractiveEval from "./pages/InteractiveEval.vue";
import Backtest from "./pages/Backtest.vue";
import StyleCrowding from "./pages/StyleCrowding.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: "/",
      component: AppLayout,
      children: [
        { path: "", component: Home },
        { path: "factors", component: Dashboard },
        { path: "backtest", component: Backtest },
        { path: "detail/:factor?", component: FactorDetail },
        { path: "families", component: Families },
        { path: "families-eval", component: FamilyEval },
        { path: "filter", component: FactorFilter },
        { path: "docs/:slug?", component: Docs },
        { path: "batch", component: BatchEval },
        { path: "style-crowding", component: StyleCrowding },
        { path: "eval", component: InteractiveEval },
      ],
    },
  ],
});

export default router;
