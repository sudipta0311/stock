// OpenAPI SHA256: 17e7d8999ae348f54f50e72245719280f0ae862f0180951293a44bdb1cafee67
export interface paths {
    "/healthz": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Healthz */
        get: operations["healthz_healthz_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/accounts": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List Accounts */
        get: operations["list_accounts_v1_accounts_get"];
        put?: never;
        /** Create Account */
        post: operations["create_account_v1_accounts_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/accounts/{account_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        post?: never;
        /** Delete Account */
        delete: operations["delete_account_v1_accounts__account_id__delete"];
        options?: never;
        head?: never;
        /** Patch Account */
        patch: operations["patch_account_v1_accounts__account_id__patch"];
        trace?: never;
    };
    "/v1/actions": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Actions */
        get: operations["get_actions_v1_actions_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/actions/{action_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        /** Patch Action */
        patch: operations["patch_action_v1_actions__action_id__patch"];
        trace?: never;
    };
    "/v1/assets": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List Assets */
        get: operations["list_assets_v1_assets_get"];
        put?: never;
        /** Create Asset */
        post: operations["create_asset_v1_assets_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/assets/quick-add": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Quick Add */
        post: operations["quick_add_v1_assets_quick_add_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/assets/{asset_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Asset */
        get: operations["get_asset_v1_assets__asset_id__get"];
        put?: never;
        post?: never;
        /** Delete Asset */
        delete: operations["delete_asset_v1_assets__asset_id__delete"];
        options?: never;
        head?: never;
        /** Patch Asset */
        patch: operations["patch_asset_v1_assets__asset_id__patch"];
        trace?: never;
    };
    "/v1/auth/google/exchange": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Exchange Google Code */
        post: operations["exchange_google_code_v1_auth_google_exchange_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/brief/{week}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Brief */
        get: operations["get_brief_v1_brief__week__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/email/unsubscribe": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Unsubscribe */
        get: operations["get_unsubscribe_v1_email_unsubscribe_get"];
        put?: never;
        /** Post Unsubscribe */
        post: operations["post_unsubscribe_v1_email_unsubscribe_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/fund-check": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Fund Check */
        get: operations["get_fund_check_v1_fund_check_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/fund-check/sips/{security_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /** Put Sip Override */
        put: operations["put_sip_override_v1_fund_check_sips__security_id__put"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/goals": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List Goals */
        get: operations["list_goals_v1_goals_get"];
        put?: never;
        /** Create Goal */
        post: operations["create_goal_v1_goals_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/goals/{goal_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Goal */
        get: operations["get_goal_v1_goals__goal_id__get"];
        put?: never;
        post?: never;
        /** Delete Goal */
        delete: operations["delete_goal_v1_goals__goal_id__delete"];
        options?: never;
        head?: never;
        /** Patch Goal */
        patch: operations["patch_goal_v1_goals__goal_id__patch"];
        trace?: never;
    };
    "/v1/highlights/dismissals": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Highlight Dismissals */
        get: operations["get_highlight_dismissals_v1_highlights_dismissals_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/highlights/{key}/dismiss": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Dismiss Highlight */
        post: operations["dismiss_highlight_v1_highlights__key__dismiss_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/households": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List Households */
        get: operations["list_households_v1_households_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/households/access": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List Access */
        get: operations["list_access_v1_households_access_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/households/access/{grantee_user_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        post?: never;
        /** Revoke Access */
        delete: operations["revoke_access_v1_households_access__grantee_user_id__delete"];
        options?: never;
        head?: never;
        /** Update Access */
        patch: operations["update_access_v1_households_access__grantee_user_id__patch"];
        trace?: never;
    };
    "/v1/households/invites": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List Invites */
        get: operations["list_invites_v1_households_invites_get"];
        put?: never;
        /** Create Invite */
        post: operations["create_invite_v1_households_invites_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/households/invites/{invite_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        post?: never;
        /** Revoke Invite */
        delete: operations["revoke_invite_v1_households_invites__invite_id__delete"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/households/join": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Join Household */
        post: operations["join_household_v1_households_join_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/households/leave": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        post?: never;
        /** Leave Household */
        delete: operations["leave_household_v1_households_leave_delete"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/imports": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List Imports */
        get: operations["list_imports_v1_imports_get"];
        put?: never;
        /** Upload Import */
        post: operations["upload_import_v1_imports_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/imports/{job_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Import */
        get: operations["get_import_v1_imports__job_id__get"];
        put?: never;
        post?: never;
        /** Discard Import */
        delete: operations["discard_import_v1_imports__job_id__delete"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/imports/{job_id}/confirm": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Confirm Import */
        post: operations["confirm_import_v1_imports__job_id__confirm_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/imports/{job_id}/items/{item_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        /** Patch Import Item */
        patch: operations["patch_import_item_v1_imports__job_id__items__item_id__patch"];
        trace?: never;
    };
    "/v1/imports/{job_id}/mapping": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Map Import Columns */
        post: operations["map_import_columns_v1_imports__job_id__mapping_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/inbox": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Inbox */
        get: operations["get_inbox_v1_inbox_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/me": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Me */
        get: operations["get_me_v1_me_get"];
        put?: never;
        post?: never;
        /** Delete Me */
        delete: operations["delete_me_v1_me_delete"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/me/checklist": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Checklist */
        get: operations["get_checklist_v1_me_checklist_get"];
        /** Put Checklist */
        put: operations["put_checklist_v1_me_checklist_put"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/me/consents": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Consents */
        get: operations["get_consents_v1_me_consents_get"];
        /** Put Consents */
        put: operations["put_consents_v1_me_consents_put"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/me/investing": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Investing */
        get: operations["get_investing_v1_me_investing_get"];
        /** Put Investing */
        put: operations["put_investing_v1_me_investing_put"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/me/profile": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /** Put Profile */
        put: operations["put_profile_v1_me_profile_put"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/me/residency": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Residency */
        get: operations["get_residency_v1_me_residency_get"];
        /** Put Residency */
        put: operations["put_residency_v1_me_residency_put"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/me/risk-answers": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /** Put Risk Answers */
        put: operations["put_risk_answers_v1_me_risk_answers_put"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/members": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List Members */
        get: operations["list_members_v1_members_get"];
        put?: never;
        /** Create Member */
        post: operations["create_member_v1_members_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/members/{member_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        post?: never;
        /** Delete Member */
        delete: operations["delete_member_v1_members__member_id__delete"];
        options?: never;
        head?: never;
        /** Patch Member */
        patch: operations["patch_member_v1_members__member_id__patch"];
        trace?: never;
    };
    "/v1/meta/risk-questions": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Risk Questions */
        get: operations["get_risk_questions_v1_meta_risk_questions_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/networth": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Networth */
        get: operations["get_networth_v1_networth_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/plan": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Plan */
        get: operations["get_plan_v1_plan_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/push/subscriptions": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Post Push Subscription */
        post: operations["post_push_subscription_v1_push_subscriptions_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/push/subscriptions/{subscription_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        post?: never;
        /** Delete Push Subscription */
        delete: operations["delete_push_subscription_v1_push_subscriptions__subscription_id__delete"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/research/ideas": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Research Ideas */
        get: operations["get_research_ideas_v1_research_ideas_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/stress": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Stress */
        get: operations["get_stress_v1_stress_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
}
export type webhooks = Record<string, never>;
export interface components {
    schemas: {
        /** AccountPatch */
        AccountPatch: {
            /** Account Type */
            account_type?: string | null;
            /** Institution */
            institution?: string | null;
            /** Joint */
            joint?: boolean | null;
            /** Masked Number */
            masked_number?: string | null;
            /** Member Id */
            member_id?: string | null;
            /** Tax Wrapper */
            tax_wrapper?: ("ask" | "ordinary" | "pension") | null;
        };
        /** AccountResponse */
        AccountResponse: {
            /** Account Type */
            account_type: string;
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /** Institution */
            institution: string;
            /** Joint */
            joint: boolean;
            /** Masked Number */
            masked_number: string | null;
            /** Member Id */
            member_id: string | null;
            /** Tax Wrapper */
            tax_wrapper?: ("ask" | "ordinary" | "pension") | null;
        };
        /** AccountWrite */
        AccountWrite: {
            /** Account Type */
            account_type: string;
            /** Institution */
            institution: string;
            /**
             * Joint
             * @default false
             */
            joint: boolean;
            /** Masked Number */
            masked_number?: string | null;
            /** Member Id */
            member_id?: string | null;
            /** Tax Wrapper */
            tax_wrapper?: ("ask" | "ordinary" | "pension") | null;
        };
        /** ActionResponse */
        ActionResponse: {
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Done By */
            done_by?: string | null;
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /** Kind */
            kind: string;
            /** Payload */
            payload: {
                [key: string]: unknown;
            };
            /** Rank */
            rank: number;
            /**
             * Status
             * @enum {string}
             */
            status: "proposed" | "done" | "dismissed";
            /** Week */
            week: string;
        };
        /** ActionStatusUpdate */
        ActionStatusUpdate: {
            /**
             * Status
             * @enum {string}
             */
            status: "done" | "dismissed";
        };
        /** AssetDetailResponse */
        AssetDetailResponse: {
            /** Account Id */
            account_id: string | null;
            /** Asset Type */
            asset_type: string;
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Details */
            details: {
                [key: string]: unknown;
            };
            /** History */
            history: components["schemas"]["AssetHistoryResponse"][];
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /** Joint */
            joint: boolean;
            /**
             * Member Id
             * Format: uuid
             */
            member_id: string;
            /**
             * Status
             * @enum {string}
             */
            status: "active" | "closed";
            /**
             * Updated At
             * Format: date-time
             */
            updated_at: string;
            /** Valuation */
            valuation: {
                [key: string]: unknown;
            } | null;
            /** Valuation Note */
            valuation_note: string | null;
            /**
             * Valuation Status
             * @enum {string}
             */
            valuation_status: "valued" | "price_needed" | "rate_needed" | "not_applicable";
            /** Value As Of */
            value_as_of: string | null;
            /** Value Cached */
            value_cached: string | null;
        };
        /** AssetHistoryResponse */
        AssetHistoryResponse: {
            /**
             * Changed At
             * Format: date-time
             */
            changed_at: string;
            /** Changed By */
            changed_by: string;
            /** Changed Fields */
            changed_fields: {
                [key: string]: {
                    [key: string]: string;
                };
            };
            /**
             * Id
             * Format: uuid
             */
            id: string;
        };
        /** AssetResponse */
        AssetResponse: {
            /** Account Id */
            account_id: string | null;
            /** Asset Type */
            asset_type: string;
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Details */
            details: {
                [key: string]: unknown;
            };
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /** Joint */
            joint: boolean;
            /**
             * Member Id
             * Format: uuid
             */
            member_id: string;
            /**
             * Status
             * @enum {string}
             */
            status: "active" | "closed";
            /**
             * Updated At
             * Format: date-time
             */
            updated_at: string;
            /** Valuation */
            valuation: {
                [key: string]: unknown;
            } | null;
            /** Valuation Note */
            valuation_note: string | null;
            /**
             * Valuation Status
             * @enum {string}
             */
            valuation_status: "valued" | "price_needed" | "rate_needed" | "not_applicable";
            /** Value As Of */
            value_as_of: string | null;
            /** Value Cached */
            value_cached: string | null;
        };
        /** BondAsset */
        BondAsset: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            asset_type: "bond";
            /** Coupon Frequency */
            coupon_frequency?: (1 | 2) | null;
            /** Coupon Pct */
            coupon_pct?: string | null;
            /**
             * Face Value
             * Format: decimal
             */
            face_value: string;
            /** Isin */
            isin?: string | null;
            /** Issuer */
            issuer?: string | null;
            /**
             * Joint
             * @default false
             */
            joint: boolean;
            /**
             * Kind
             * @enum {string}
             */
            kind: "gsec" | "tbill" | "rbi_frsb" | "corporate" | "tax_free" | "bond_54ec";
            /** Label */
            label?: string | null;
            maturity: string;
            /** Member Id */
            member_id?: string | null;
            /** Notes */
            notes?: string | null;
        };
        /** Brief */
        Brief: {
            /** Actions */
            actions: components["schemas"]["BriefAction"][];
            /**
             * As Of
             * Format: date
             */
            as_of: string;
            /** Disclaimer */
            disclaimer: string;
            /** Household Header */
            household_header?: string | null;
            /** Manage Alerts Url */
            manage_alerts_url: string;
            /**
             * No Llm Numbers
             * @default true
             * @constant
             */
            no_llm_numbers: true;
            /** Plan Health */
            plan_health: number;
            /** Postal Address */
            postal_address: string;
            /** Preheader */
            preheader: string;
            /** Sections */
            sections: {
                [key: string]: unknown;
            };
            /**
             * Status
             * @enum {string}
             */
            status: "ACTIONS" | "NONE";
            /** Subject */
            subject: string;
            /** Unsubscribe Url */
            unsubscribe_url: string;
            /** Week */
            week: string;
            /** Why Received */
            why_received: string;
        };
        /** BriefAction */
        BriefAction: {
            /** Amount */
            amount: string | null;
            /** Deeplink */
            deeplink: string;
            /** Kind */
            kind: string;
            /**
             * Member Name
             * @default You
             */
            member_name: string;
            /**
             * Member Relationship
             * @default self
             */
            member_relationship: string;
            /** Rank */
            rank: number;
            /** Title */
            title: string;
            /** Why */
            why: string;
        };
        /** ChecklistWrite */
        ChecklistWrite: {
            /** Categories */
            categories: string[];
        };
        /** CompletenessResponse */
        CompletenessResponse: {
            /** Done */
            done: string[];
            /** Pct */
            pct: string;
            /** Ticked */
            ticked: string[];
        };
        /**
         * Compounding
         * @enum {string}
         */
        Compounding: "simple" | "monthly" | "quarterly" | "half_yearly" | "yearly";
        /** ConsentValues */
        ConsentValues: {
            /**
             * Push
             * @default false
             */
            push: boolean;
            /**
             * Reminder Email
             * @default false
             */
            reminder_email: boolean;
            /**
             * Urgent Alerts
             * @default false
             */
            urgent_alerts: boolean;
            /**
             * Weekly Email
             * @default false
             */
            weekly_email: boolean;
        };
        /** DebtInsight */
        DebtInsight: {
            /** Debt Pct Including Locked */
            debt_pct_including_locked: string;
            /**
             * Locked Debt Amount
             * Format: decimal
             */
            locked_debt_amount: string;
            /**
             * Message Key
             * @constant
             */
            message_key: "debt_share";
        };
        /** EPFAsset */
        EPFAsset: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            asset_type: "epf";
            /**
             * Balance
             * Format: decimal
             */
            balance: string;
            balance_as_of?: string | null;
            /**
             * Joint
             * @default false
             */
            joint: boolean;
            /** Label */
            label?: string | null;
            /** Member Id */
            member_id?: string | null;
            /** Monthly Employee */
            monthly_employee?: string | null;
            /** Monthly Employer */
            monthly_employer?: string | null;
            /** Notes */
            notes?: string | null;
            /** Vpf Monthly */
            vpf_monthly?: string | null;
        };
        /** FinancialContextFields */
        FinancialContextFields: {
            /** Annual Income */
            annual_income?: string | null;
            /** City Tier */
            city_tier?: (1 | 2 | 3) | null;
            /** Dependants */
            dependants?: number | null;
            /** Employment Type */
            employment_type?: ("salaried" | "self_employed" | "business" | "retired" | "other") | null;
            /** Income Stability */
            income_stability?: ("high" | "medium" | "low") | null;
            /** Monthly Expenses */
            monthly_expenses?: string | null;
        };
        /** FixedDepositAsset */
        FixedDepositAsset: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            asset_type: "fd";
            /** Bank */
            bank: string;
            /** @default quarterly */
            compounding: components["schemas"]["Compounding"];
            /**
             * Joint
             * @default false
             */
            joint: boolean;
            /** Label */
            label?: string | null;
            maturity: string;
            /** Member Id */
            member_id?: string | null;
            /** Notes */
            notes?: string | null;
            /** @default cumulative */
            payout: components["schemas"]["PayoutType"];
            /**
             * Principal
             * Format: decimal
             */
            principal: string;
            /**
             * Rate Pct
             * Format: decimal
             */
            rate_pct: string;
            start?: string | null;
        };
        /** FundCheckResponse */
        FundCheckResponse: {
            /** Checks */
            checks: {
                [key: string]: unknown;
            }[];
            /** Details */
            details: {
                [key: string]: unknown;
            };
            /** Early Warning */
            early_warning: {
                [key: string]: unknown;
            } | null;
            /** Status */
            status: string;
            /** Summary Key */
            summary_key: string;
        };
        /** FundSipOverride */
        FundSipOverride: {
            /**
             * Member Id
             * Format: uuid
             */
            member_id: string;
            /** Monthly Amount Inr */
            monthly_amount_inr: number | string;
        };
        /** Goal */
        Goal: {
            /**
             * Currency
             * @enum {string}
             */
            currency: "INR" | "DKK" | "EUR" | "USD";
            goal_type: components["schemas"]["GoalType"];
            /** Member Id */
            member_id?: string | null;
            /** Name */
            name: string;
            /** Priority */
            priority: number;
            /**
             * Target Amount
             * Format: decimal
             */
            target_amount: string;
            /** Target Year */
            target_year: number;
        };
        /** GoalResponse */
        GoalResponse: {
            /**
             * Currency
             * @enum {string}
             */
            currency: "INR" | "DKK" | "EUR" | "USD";
            /** Goal Type */
            goal_type: string;
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /** Member Id */
            member_id: string | null;
            /** Name */
            name: string;
            /** Priority */
            priority: number;
            /**
             * Target Amount
             * Format: decimal
             */
            target_amount: string;
            /** Target Year */
            target_year: number;
        };
        /**
         * GoalType
         * @enum {string}
         */
        GoalType: "emergency" | "education_india" | "education_abroad" | "home" | "retirement" | "wealth" | "other";
        /** GoldAsset */
        GoldAsset: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            asset_type: "gold";
            /** Grams */
            grams?: number | string | null;
            issue?: string | null;
            /** Issue Price Per Gram */
            issue_price_per_gram?: string | null;
            /**
             * Joint
             * @default false
             */
            joint: boolean;
            /**
             * Kind
             * @enum {string}
             */
            kind: "sgb" | "etf" | "mf" | "physical" | "digital";
            /** Label */
            label?: string | null;
            /** Member Id */
            member_id?: string | null;
            /** Notes */
            notes?: string | null;
            /** Purity */
            purity?: (24 | 22 | 18) | null;
            /** Purpose */
            purpose?: ("jewellery" | "investment") | null;
            /** Series */
            series?: string | null;
            /** Units */
            units?: number | string | null;
        };
        /** GoogleExchangeResponse */
        GoogleExchangeResponse: {
            /** Expires In */
            expires_in: number;
            /** Id Token */
            id_token: string;
        };
        /** HTTPValidationError */
        HTTPValidationError: {
            /** Detail */
            detail?: components["schemas"]["ValidationError"][];
        };
        /** HealthInsuranceAsset */
        HealthInsuranceAsset: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            asset_type: "health_insurance";
            /**
             * Cover
             * Format: decimal
             */
            cover: string;
            /** Insurer */
            insurer: string;
            /**
             * Joint
             * @default false
             */
            joint: boolean;
            /** Label */
            label?: string | null;
            /** Member Id */
            member_id?: string | null;
            /** Members Covered */
            members_covered?: string[] | null;
            /** Notes */
            notes?: string | null;
            /** Renewal Month */
            renewal_month: number;
            /** Top Up Cover */
            top_up_cover?: string | null;
        };
        /** HealthResponse */
        HealthResponse: {
            /** Db */
            db?: ("ok" | "error") | null;
            /**
             * Status
             * @constant
             */
            status: "ok";
            /** Version */
            version: string;
        };
        /**
         * Highlight
         * @description Small, plain-language item suitable for a screen summary.
         */
        Highlight: {
            /** Body */
            body: string;
            /** Key */
            key: string;
            /** Learn More Key */
            learn_more_key: string;
            /**
             * Screen
             * @enum {string}
             */
            screen: "home" | "plan" | "ideas" | "money";
            /**
             * Severity
             * @enum {string}
             */
            severity: "info" | "attention";
            /** Title */
            title: string;
        };
        /** HighlightDismissalResponse */
        HighlightDismissalResponse: {
            /**
             * Dismissed At
             * Format: date-time
             */
            dismissed_at: string;
            /** Fingerprint */
            fingerprint: string;
            /** Key */
            key: string;
        };
        /** HoldingValuationResponse */
        HoldingValuationResponse: {
            /** Currency */
            currency: string;
            /** Fx As Of */
            fx_as_of: string | null;
            /** Fx Missing */
            fx_missing: boolean;
            /** Name */
            name: string;
            /** Price As Of */
            price_as_of: string | null;
            /** Price Source */
            price_source: string | null;
            /** Quantity */
            quantity: string;
            /**
             * Security Id
             * Format: uuid
             */
            security_id: string;
            /** Security Type */
            security_type: string;
            /**
             * Status
             * @enum {string}
             */
            status: "priced" | "statement_value_used" | "price_unavailable" | "fx_unavailable";
            /** Symbol */
            symbol: string;
            /** Value */
            value: string | null;
            /** Value Usd */
            value_usd: string | null;
        };
        /** HouseholdAccessPatch */
        HouseholdAccessPatch: {
            /**
             * Role
             * @enum {string}
             */
            role: "partner" | "viewer";
        };
        /** HouseholdAccessResponse */
        HouseholdAccessResponse: {
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Display Name */
            display_name: string | null;
            /**
             * Member Id
             * Format: uuid
             */
            member_id: string;
            /** Revoked At */
            revoked_at: string | null;
            /**
             * Role
             * @enum {string}
             */
            role: "partner" | "viewer";
            /**
             * User Id
             * Format: uuid
             */
            user_id: string;
        };
        /** HouseholdJoinResponse */
        HouseholdJoinResponse: {
            /** Merge Counts */
            merge_counts?: {
                [key: string]: number;
            } | null;
            /** Overwritten Fields */
            overwritten_fields?: string[];
            /**
             * Owner Id
             * Format: uuid
             */
            owner_id: string;
        };
        /** HouseholdSummary */
        HouseholdSummary: {
            /** Member Count */
            member_count: number;
            /**
             * Owner Id
             * Format: uuid
             */
            owner_id: string;
            /** Owner Name */
            owner_name: string | null;
            /**
             * Role
             * @enum {string}
             */
            role: "owner" | "partner" | "viewer";
        };
        /** ImportConfirmationRequest */
        ImportConfirmationRequest: {
            /** Tax Wrapper */
            tax_wrapper?: ("ask" | "ordinary" | "pension") | null;
        };
        /** ImportConfirmationResponse */
        ImportConfirmationResponse: {
            /** Counts */
            counts: {
                [key: string]: number;
            };
            /**
             * Job Id
             * Format: uuid
             */
            job_id: string;
            /** Net Worth */
            net_worth: string;
            /**
             * Status
             * @constant
             */
            status: "confirmed";
        };
        /** ImportDetailResponse */
        ImportDetailResponse: {
            /** Account Id */
            account_id?: string | null;
            /** Broker */
            broker?: string | null;
            /**
             * Cash Flow Count
             * @default 0
             */
            cash_flow_count: number;
            /** Confirmed At */
            confirmed_at: string | null;
            /** Counts */
            counts: {
                [key: string]: number;
            };
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /**
             * Currency Default
             * @default USD
             * @enum {string}
             */
            currency_default: "INR" | "USD" | "EUR" | "DKK" | "GBP" | "SEK";
            /** Doc Type */
            doc_type: string;
            /** Error Code */
            error_code: string | null;
            /** Headers */
            headers?: string[] | null;
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /** Items */
            items: components["schemas"]["ImportItemResponse"][];
            /**
             * Market Default
             * @default US
             * @enum {string}
             */
            market_default: "IN" | "US" | "DK" | "EU" | "GB";
            /** Member Id */
            member_id?: string | null;
            /** Samples */
            samples?: string[][];
            /** Source */
            source: ("nsdl" | "cdsl" | "cams_kfin" | "broker") | null;
            /** Statement Date */
            statement_date: string | null;
            /**
             * Status
             * @enum {string}
             */
            status: "ready" | "needs_mapping" | "confirmed" | "failed" | "discarded";
            /** Suggested Mapping */
            suggested_mapping?: {
                [key: string]: string | null;
            };
            /** Tax Wrapper */
            tax_wrapper?: ("ask" | "ordinary" | "pension") | null;
            /**
             * Tax Wrapper Question
             * @default false
             */
            tax_wrapper_question: boolean;
        };
        /** ImportItemResponse */
        ImportItemResponse: {
            /** Account Ref */
            account_ref: string | null;
            /** Action */
            action?: string | null;
            /** Amount */
            amount?: string | null;
            /** Currency */
            currency?: string | null;
            /** Diff */
            diff: {
                [key: string]: unknown;
            } | null;
            /** Exchange */
            exchange?: string | null;
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /** Isin */
            isin: string;
            /** Kind */
            kind: string;
            /** Market */
            market?: string | null;
            /** Match */
            match: {
                [key: string]: string | null;
            };
            /** Name */
            name: string;
            /** Quantity */
            quantity: string;
            /** Reason */
            reason?: string | null;
            /** Sleeve */
            sleeve: string | null;
            /** Statement Value */
            statement_value: string | null;
            /** Status */
            status: string;
            /** Symbol */
            symbol?: string | null;
        };
        /** ImportJobResponse */
        ImportJobResponse: {
            /** Account Id */
            account_id?: string | null;
            /** Confirmed At */
            confirmed_at: string | null;
            /** Counts */
            counts: {
                [key: string]: number;
            };
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Doc Type */
            doc_type: string;
            /** Error Code */
            error_code: string | null;
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /** Member Id */
            member_id?: string | null;
            /** Source */
            source: ("nsdl" | "cdsl" | "cams_kfin" | "broker") | null;
            /** Statement Date */
            statement_date: string | null;
            /**
             * Status
             * @enum {string}
             */
            status: "ready" | "needs_mapping" | "confirmed" | "failed" | "discarded";
        };
        /** ImportMappingRequest */
        ImportMappingRequest: {
            /**
             * Broker
             * @default other
             * @enum {string}
             */
            broker: "nordnet" | "saxo" | "danske_bank" | "ibkr" | "vested" | "indmoney" | "other";
            /** Columns */
            columns: {
                [key: string]: string;
            };
            /**
             * Currency Default
             * @default USD
             * @enum {string}
             */
            currency_default: "INR" | "USD" | "EUR" | "DKK" | "GBP" | "SEK";
            /**
             * Kind
             * @enum {string}
             */
            kind: "holdings" | "tradebook";
            /**
             * Market Default
             * @default US
             * @enum {string}
             */
            market_default: "IN" | "US" | "DK" | "EU" | "GB";
            /** Save As */
            save_as?: string | null;
        };
        /** ImportUploadResponse */
        ImportUploadResponse: {
            /** Counts */
            counts: {
                [key: string]: number;
            };
            /**
             * Job Id
             * Format: uuid
             */
            job_id: string;
            /**
             * Source
             * @enum {string}
             */
            source: "nsdl" | "cdsl" | "cams_kfin" | "broker";
            /** Statement Date */
            statement_date: string | null;
            /**
             * Status
             * @enum {string}
             */
            status: "ready" | "needs_mapping";
            /** Warnings */
            warnings: string[];
        };
        /** InboxItem */
        InboxItem: {
            /** Id */
            id: string;
            /** Kind */
            kind: string;
            /** Sent At */
            sent_at?: string | null;
            /** Status */
            status?: string | null;
            /** Subject */
            subject: string;
            /** Week */
            week?: string | null;
        };
        /** InboxResponse */
        InboxResponse: {
            /** Items */
            items: components["schemas"]["InboxItem"][];
        };
        /**
         * InvestingPrefs
         * @description User-controlled investing constraints used by the deterministic policy engine.
         */
        "InvestingPrefs-Input": {
            /**
             * Ask Used This Fy Dkk
             * Format: decimal
             * @default 0
             */
            ask_used_this_fy_dkk: string;
            /** Ask Value Start Of Year Dkk */
            ask_value_start_of_year_dkk?: string | null;
            /** Birth Year */
            birth_year?: number | null;
            /**
             * Broker
             * @default nordnet
             * @enum {string}
             */
            broker: "nordnet" | "saxo" | "danske_bank" | "other";
            /** Existing Sips */
            existing_sips?: components["schemas"]["InvestingSip"][];
            /**
             * Has Danish Pension Account
             * @default false
             */
            has_danish_pension_account: boolean;
            /**
             * Has Indian Nri Share Account
             * @default false
             */
            has_indian_nri_share_account: boolean;
            /**
             * Has Lrs Broker
             * @default false
             */
            has_lrs_broker: boolean;
            /**
             * Lrs Used This Fy
             * Format: decimal
             * @default 0
             */
            lrs_used_this_fy: string;
            /**
             * Monthly Sip Capacity
             * Format: decimal
             * @default 0
             */
            monthly_sip_capacity: string;
            /**
             * New Money This Week
             * Format: decimal
             * @default 0
             */
            new_money_this_week: string;
            /**
             * Prefers No Paperwork
             * @default false
             */
            prefers_no_paperwork: boolean;
            /**
             * Satellite Pct
             * Format: decimal
             * @default 20
             */
            satellite_pct: string;
            /**
             * Us Stock Picks Opt In
             * @default false
             */
            us_stock_picks_opt_in: boolean;
            /**
             * Us Stock Picks Pct
             * Format: decimal
             * @default 10
             */
            us_stock_picks_pct: string;
        };
        /**
         * InvestingPrefs
         * @description User-controlled investing constraints used by the deterministic policy engine.
         */
        "InvestingPrefs-Output": {
            /**
             * Ask Used This Fy Dkk
             * Format: decimal
             * @default 0.00
             */
            ask_used_this_fy_dkk: string;
            /** Ask Value Start Of Year Dkk */
            ask_value_start_of_year_dkk?: string | null;
            /** Birth Year */
            birth_year?: number | null;
            /**
             * Broker
             * @default nordnet
             * @enum {string}
             */
            broker: "nordnet" | "saxo" | "danske_bank" | "other";
            /** Existing Sips */
            existing_sips?: components["schemas"]["InvestingSip"][];
            /**
             * Has Danish Pension Account
             * @default false
             */
            has_danish_pension_account: boolean;
            /**
             * Has Indian Nri Share Account
             * @default false
             */
            has_indian_nri_share_account: boolean;
            /**
             * Has Lrs Broker
             * @default false
             */
            has_lrs_broker: boolean;
            /**
             * Lrs Used This Fy
             * Format: decimal
             * @default 0.00
             */
            lrs_used_this_fy: string;
            /**
             * Monthly Sip Capacity
             * Format: decimal
             * @default 0.00
             */
            monthly_sip_capacity: string;
            /**
             * New Money This Week
             * Format: decimal
             * @default 0.00
             */
            new_money_this_week: string;
            /**
             * Prefers No Paperwork
             * @default false
             */
            prefers_no_paperwork: boolean;
            /**
             * Satellite Pct
             * Format: decimal
             * @default 20
             */
            satellite_pct: string;
            /**
             * Us Stock Picks Opt In
             * @default false
             */
            us_stock_picks_opt_in: boolean;
            /**
             * Us Stock Picks Pct
             * Format: decimal
             * @default 10
             */
            us_stock_picks_pct: string;
        };
        /**
         * InvestingSip
         * @description One existing monthly investment commitment.
         */
        InvestingSip: {
            /** Instrument Category */
            instrument_category: string;
            /**
             * Monthly Amount
             * Format: decimal
             */
            monthly_amount: string;
            /**
             * Sleeve
             * @enum {string}
             */
            sleeve: "equity_india" | "equity_global" | "debt" | "gold" | "cash";
        };
        /** InviteCreate */
        InviteCreate: {
            /** Email */
            email: string;
            /** Member Id */
            member_id?: string | null;
            /**
             * Role
             * @default partner
             * @enum {string}
             */
            role: "partner" | "viewer";
            /**
             * Send Email
             * @default false
             */
            send_email: boolean;
        };
        /** InviteCreatedResponse */
        InviteCreatedResponse: {
            /** Accepted At */
            accepted_at: string | null;
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Email */
            email: string;
            /**
             * Email Sent
             * @default false
             */
            email_sent: boolean;
            /**
             * Expires At
             * Format: date-time
             */
            expires_at: string;
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /** Member Id */
            member_id: string | null;
            /** Revoked At */
            revoked_at: string | null;
            /**
             * Role
             * @enum {string}
             */
            role: "partner" | "viewer";
            /** Share Link */
            share_link: string;
        };
        /** InviteJoin */
        InviteJoin: {
            /**
             * Bring My Data
             * @default false
             */
            bring_my_data: boolean;
            /** Token */
            token: string;
        };
        /** InviteResponse */
        InviteResponse: {
            /** Accepted At */
            accepted_at: string | null;
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Email */
            email: string;
            /**
             * Expires At
             * Format: date-time
             */
            expires_at: string;
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /** Member Id */
            member_id: string | null;
            /** Revoked At */
            revoked_at: string | null;
            /**
             * Role
             * @enum {string}
             */
            role: "partner" | "viewer";
        };
        /** LiquiditySummary */
        LiquiditySummary: {
            /**
             * Instant
             * Format: decimal
             */
            instant: string;
            /**
             * Locked
             * Format: decimal
             */
            locked: string;
            /**
             * Within 1Y
             * Format: decimal
             */
            within_1y: string;
        };
        /** LoanAsset */
        LoanAsset: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            asset_type: "loan";
            /** Emi */
            emi?: string | null;
            end?: string | null;
            /**
             * Joint
             * @default false
             */
            joint: boolean;
            /** Label */
            label?: string | null;
            /**
             * Loan Kind
             * @enum {string}
             */
            loan_kind: "home" | "car" | "personal" | "education" | "credit_card" | "gold" | "other";
            /** Member Id */
            member_id?: string | null;
            /** Notes */
            notes?: string | null;
            /**
             * Outstanding
             * Format: decimal
             */
            outstanding: string;
            /**
             * Rate Pct
             * Format: decimal
             */
            rate_pct: string;
        };
        /** MaturityEvent */
        MaturityEvent: {
            /** Amount */
            amount: string | null;
            /**
             * Asset Id
             * Format: uuid
             */
            asset_id: string;
            /**
             * Date
             * Format: date
             */
            date: string;
            /** Kind */
            kind: string;
            /** Label */
            label: string;
        };
        /** MeResponse */
        MeResponse: {
            /** Checklist */
            checklist: string[];
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Display Name */
            display_name: string | null;
            /** Email */
            email: string;
            /**
             * Household Owner Id
             * Format: uuid
             */
            household_owner_id: string;
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /**
             * Me Member Id
             * Format: uuid
             */
            me_member_id: string;
            /** Members */
            members?: components["schemas"]["MemberResponse"][];
            profile: components["schemas"]["ProfileResponse"] | null;
            residency: components["schemas"]["Residency"];
            /**
             * Role
             * @enum {string}
             */
            role: "owner" | "partner" | "viewer";
        };
        /** MemberPatch */
        MemberPatch: {
            /** Display Name */
            display_name?: string | null;
            investing?: components["schemas"]["InvestingPrefs-Input"] | null;
            /** Is Self */
            is_self?: boolean | null;
            /** Relationship */
            relationship?: ("self" | "spouse" | "child" | "parent" | "other") | null;
            residency?: components["schemas"]["ResidencyInput"] | null;
            tax?: components["schemas"]["MemberTax-Input"] | null;
        };
        /** MemberResponse */
        MemberResponse: {
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Display Name */
            display_name: string;
            /**
             * Id
             * Format: uuid
             */
            id: string;
            investing: components["schemas"]["InvestingPrefs-Output"];
            /** Is Self */
            is_self: boolean;
            /** Relationship */
            relationship: string;
            residency: components["schemas"]["Residency"];
            tax: components["schemas"]["MemberTax-Output"];
        };
        /**
         * MemberTax
         * @description Tax and earning facts attached to one household member.
         */
        "MemberTax-Input": {
            /** Annual Income */
            annual_income?: string | null;
            /**
             * Annual Income Currency
             * @default INR
             * @enum {string}
             */
            annual_income_currency: "INR" | "DKK" | "EUR" | "USD";
            /** Birth Year */
            birth_year?: number | null;
            /**
             * Employment Type
             * @default other
             * @enum {string}
             */
            employment_type: "salaried" | "self_employed" | "business" | "retired" | "other";
            /**
             * Fy Realised Equity Ltcg
             * Format: decimal
             * @default 0
             */
            fy_realised_equity_ltcg: string;
            /**
             * Fy Realised Equity Stcg
             * Format: decimal
             * @default 0
             */
            fy_realised_equity_stcg: string;
            /**
             * Surcharge Pct
             * Format: decimal
             * @default 0
             */
            surcharge_pct: string;
            tax_regime?: components["schemas"]["TaxRegime"] | null;
            /** Tax Slab Pct */
            tax_slab_pct?: string | null;
        };
        /**
         * MemberTax
         * @description Tax and earning facts attached to one household member.
         */
        "MemberTax-Output": {
            /** Annual Income */
            annual_income?: string | null;
            /**
             * Annual Income Currency
             * @default INR
             * @enum {string}
             */
            annual_income_currency: "INR" | "DKK" | "EUR" | "USD";
            /** Birth Year */
            birth_year?: number | null;
            /**
             * Employment Type
             * @default other
             * @enum {string}
             */
            employment_type: "salaried" | "self_employed" | "business" | "retired" | "other";
            /**
             * Fy Realised Equity Ltcg
             * Format: decimal
             * @default 0.00
             */
            fy_realised_equity_ltcg: string;
            /**
             * Fy Realised Equity Stcg
             * Format: decimal
             * @default 0.00
             */
            fy_realised_equity_stcg: string;
            /**
             * Surcharge Pct
             * Format: decimal
             * @default 0
             */
            surcharge_pct: string;
            tax_regime?: components["schemas"]["TaxRegime"] | null;
            /** Tax Slab Pct */
            tax_slab_pct?: string | null;
        };
        /** MemberWrite */
        MemberWrite: {
            /** Display Name */
            display_name: string;
            investing?: components["schemas"]["InvestingPrefs-Input"];
            /**
             * Is Self
             * @default false
             */
            is_self: boolean;
            /**
             * Relationship
             * @enum {string}
             */
            relationship: "self" | "spouse" | "child" | "parent" | "other";
            residency?: components["schemas"]["ResidencyInput"];
            tax?: components["schemas"]["MemberTax-Input"];
        };
        /**
         * MonthYear
         * @description A validated calendar year and month without a specific day.
         */
        MonthYear: {
            /** Month */
            month: number;
            /** Year */
            year: number;
        };
        /** NPSAsset */
        NPSAsset: {
            /** Alternate Pct */
            alternate_pct?: string | null;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            asset_type: "nps";
            /** Corporate Debt Pct */
            corporate_debt_pct?: string | null;
            /** Equity Pct */
            equity_pct?: string | null;
            /** Govt Debt Pct */
            govt_debt_pct?: string | null;
            /**
             * Joint
             * @default false
             */
            joint: boolean;
            /** Label */
            label?: string | null;
            /** Member Id */
            member_id?: string | null;
            /** Notes */
            notes?: string | null;
            /**
             * Tier1 Balance
             * Format: decimal
             */
            tier1_balance: string;
            /** Tier2 Balance */
            tier2_balance?: string | null;
        };
        /** NetworthResponse */
        NetworthResponse: {
            /**
             * As Of
             * Format: date
             */
            as_of: string;
            /**
             * Assets Total
             * Format: decimal
             */
            assets_total: string;
            /** By Sleeve */
            by_sleeve: components["schemas"]["SleeveSummary"][];
            completeness: components["schemas"]["CompletenessResponse"];
            debt_insight: components["schemas"]["DebtInsight"];
            /** Holdings */
            holdings: components["schemas"]["HoldingValuationResponse"][];
            /**
             * Investable Total
             * Format: decimal
             */
            investable_total: string;
            /**
             * Liabilities Total
             * Format: decimal
             */
            liabilities_total: string;
            liquidity_ladder: components["schemas"]["LiquiditySummary"];
            /** Market Data As Of */
            market_data_as_of: string | null;
            /** Maturity Calendar */
            maturity_calendar: components["schemas"]["MaturityEvent"][];
            /** Net Worth */
            net_worth: string;
            /** Unvalued Assets */
            unvalued_assets: components["schemas"]["UnvaluedAssetResponse"][];
        };
        /** PPFAsset */
        PPFAsset: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            asset_type: "ppf";
            /**
             * Balance
             * Format: decimal
             */
            balance: string;
            balance_as_of?: string | null;
            /** Held At */
            held_at?: string | null;
            /**
             * Joint
             * @default false
             */
            joint: boolean;
            /** Label */
            label?: string | null;
            /** Member Id */
            member_id?: string | null;
            /** Notes */
            notes?: string | null;
            /** Opened Fy Start Year */
            opened_fy_start_year: number;
            /** Yearly Contribution */
            yearly_contribution?: string | null;
        };
        /**
         * PayoutType
         * @enum {string}
         */
        PayoutType: "cumulative" | "monthly" | "quarterly" | "half_yearly" | "yearly";
        /**
         * PlanMember
         * @description Typed household-member details exposed to the PWA and API.
         */
        PlanMember: {
            /** Eligibility */
            eligibility?: {
                [key: string]: boolean;
            };
            /** Eligibility Conditions */
            eligibility_conditions?: {
                [key: string]: string;
            };
            /** Eligibility Status */
            eligibility_status?: {
                [key: string]: "allowed" | "conditional" | "not_allowed";
            };
            /** Foundation */
            foundation?: {
                [key: string]: unknown;
            };
            /** Id */
            id: string;
            /** Limits Room */
            limits_room?: {
                [key: string]: string | null;
            };
            /** Liquidity */
            liquidity?: {
                [key: string]: string;
            };
            /** Locked */
            locked?: {
                [key: string]: string;
            };
            /** Ltcg Harvest */
            ltcg_harvest?: {
                [key: string]: unknown;
            };
            /** Name */
            name: string;
            /**
             * Net Worth
             * @default 0.00
             */
            net_worth: string;
            /** Notes */
            notes?: string[];
            /**
             * Relationship
             * @enum {string}
             */
            relationship: "self" | "spouse" | "child" | "parent" | "other";
            residency?: components["schemas"]["Residency"];
            /** Sleeves */
            sleeves?: {
                [key: string]: string;
            };
        };
        /** PostOfficeAsset */
        PostOfficeAsset: {
            /**
             * Amount
             * Format: decimal
             */
            amount: string;
            /** Annual Contribution */
            annual_contribution?: string | null;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            asset_type: "po_scheme";
            balance_as_of?: string | null;
            /**
             * Joint
             * @default false
             */
            joint: boolean;
            /** Label */
            label?: string | null;
            /** Member Id */
            member_id?: string | null;
            /** Notes */
            notes?: string | null;
            purchase: string;
            /** Rate Pct */
            rate_pct?: string | null;
            /**
             * Scheme
             * @enum {string}
             */
            scheme: "nsc" | "kvp" | "scss" | "mis" | "td" | "rd" | "ssy";
            /** Td Years */
            td_years?: (1 | 2 | 3 | 5) | null;
        };
        /** ProfileResponse */
        ProfileResponse: {
            /** City Tier */
            city_tier: (1 | 2 | 3) | null;
            financial_context: components["schemas"]["FinancialContextFields"] | null;
            /** Language */
            language: string;
            residency: components["schemas"]["Residency"];
            /** Risk Answers */
            risk_answers: {
                [key: string]: string;
            } | null;
            /** Risk Profile */
            risk_profile: string | null;
            tax_regime: components["schemas"]["TaxRegime"] | null;
            /** Tax Slab Pct */
            tax_slab_pct: string | null;
            /** Timezone */
            timezone: string;
        };
        /** ProfileUpsert */
        ProfileUpsert: {
            /** Annual Income */
            annual_income?: string | null;
            /** City Tier */
            city_tier?: (1 | 2 | 3) | null;
            /** Dependants */
            dependants?: number | null;
            /** Employment Type */
            employment_type?: ("salaried" | "self_employed" | "business" | "retired" | "other") | null;
            /** Income Stability */
            income_stability?: ("high" | "medium" | "low") | null;
            /** Language */
            language?: string | null;
            /** Monthly Expenses */
            monthly_expenses?: string | null;
            tax_regime: components["schemas"]["TaxRegime"];
            /** Tax Slab Pct */
            tax_slab_pct?: string | null;
            /** Timezone */
            timezone?: string | null;
        };
        /** PropertyAsset */
        PropertyAsset: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            asset_type: "property";
            /**
             * Estimated Value
             * Format: decimal
             */
            estimated_value: string;
            /**
             * Joint
             * @default false
             */
            joint: boolean;
            /** Label */
            label?: string | null;
            /** Linked Loan Id */
            linked_loan_id?: string | null;
            /** Member Id */
            member_id?: string | null;
            /** Monthly Rent */
            monthly_rent?: string | null;
            /** Notes */
            notes?: string | null;
            /** Property Kind */
            property_kind: string;
        };
        /** PushSubscriptionInput */
        PushSubscriptionInput: {
            /**
             * Endpoint
             * Format: uri
             */
            endpoint: string;
            /** Keys */
            keys: {
                [key: string]: string;
            };
        };
        /** PushSubscriptionResponse */
        PushSubscriptionResponse: {
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Endpoint */
            endpoint: string;
            /**
             * Id
             * Format: uuid
             */
            id: string;
        };
        /** QuickAddResponse */
        QuickAddResponse: {
            /** Asset Type */
            asset_type: string;
            /**
             * Confidence
             * @enum {string}
             */
            confidence: "high" | "medium" | "low";
            /** Draft */
            draft: {
                [key: string]: unknown;
            };
            /** Missing */
            missing: string[];
        };
        /** RecurringDepositAsset */
        RecurringDepositAsset: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            asset_type: "rd";
            /** Institution */
            institution: string;
            /**
             * Joint
             * @default false
             */
            joint: boolean;
            /** Label */
            label?: string | null;
            maturity: string;
            /** Member Id */
            member_id?: string | null;
            /**
             * Monthly Amount
             * Format: decimal
             */
            monthly_amount: string;
            /** Notes */
            notes?: string | null;
            /**
             * Rate Pct
             * Format: decimal
             */
            rate_pct: string;
            start?: string | null;
        };
        /** ResearchIdeaSummary */
        ResearchIdeaSummary: {
            /** Confidence */
            confidence: string;
            /** Dossier Status */
            dossier_status: string;
            /**
             * Market
             * @enum {string}
             */
            market: "IN" | "US";
            /**
             * Price
             * Format: decimal
             */
            price: string;
            /** Quant Decile */
            quant_decile: number;
            /** Sector */
            sector: string;
            /** Security Id */
            security_id: string;
            /** Symbol */
            symbol: string;
            /** Valuation Percentile */
            valuation_percentile: string;
        };
        /** ResearchIdeasResponse */
        ResearchIdeasResponse: {
            /** Ideas */
            ideas: components["schemas"]["ResearchIdeaSummary"][];
            /** Week */
            week: string;
        };
        /**
         * Residency
         * @description Tax and country profile used by residency-aware policy decisions.
         */
        Residency: {
            /**
             * Country
             * @default IN
             * @enum {string}
             */
            country: "IN" | "DK" | "OTHER";
            /** Planned Return Year */
            planned_return_year?: number | null;
            /**
             * Settle Plan
             * @default undecided
             * @enum {string}
             */
            settle_plan: "india" | "abroad" | "undecided";
            /**
             * Status
             * @default RI
             * @enum {string}
             */
            status: "RI" | "NRI" | "RNOR";
            /**
             * Us Person
             * @default false
             */
            us_person: boolean;
        };
        /**
         * ResidencyInput
         * @description Strict API input limited to the currently supported residence choices.
         */
        ResidencyInput: {
            /**
             * Country
             * @default IN
             * @enum {string}
             */
            country: "IN" | "DK";
            /** Planned Return Year */
            planned_return_year?: number | null;
            /**
             * Settle Plan
             * @default undecided
             * @enum {string}
             */
            settle_plan: "india" | "abroad" | "undecided";
            /**
             * Status
             * @default RI
             * @enum {string}
             */
            status: "RI" | "NRI" | "RNOR";
        };
        /** RiskAnswersRequest */
        RiskAnswersRequest: {
            /** Answers */
            answers: {
                [key: string]: string;
            };
            /**
             * Emergency Ok
             * @default true
             */
            emergency_ok: boolean;
        };
        /** RiskOptionResponse */
        RiskOptionResponse: {
            /** Id */
            id: string;
            /** Text */
            text: string;
        };
        /** RiskQuestionResponse */
        RiskQuestionResponse: {
            /** Id */
            id: string;
            /** Options */
            options: components["schemas"]["RiskOptionResponse"][];
            /** Text */
            text: string;
        };
        /** RiskResultResponse */
        RiskResultResponse: {
            /** Capacity */
            capacity: string;
            /** Final */
            final: string;
            /** Limited By */
            limited_by: ("tolerance" | "capacity") | null;
            /** Tolerance */
            tolerance: string;
        };
        /** SavingsAsset */
        SavingsAsset: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            asset_type: "savings";
            /**
             * Balance
             * Format: decimal
             */
            balance: string;
            /** Bank */
            bank: string;
            /**
             * Joint
             * @default false
             */
            joint: boolean;
            /** Label */
            label?: string | null;
            /** Member Id */
            member_id?: string | null;
            /** Notes */
            notes?: string | null;
        };
        /** SleeveSummary */
        SleeveSummary: {
            /**
             * Amount
             * Format: decimal
             */
            amount: string;
            /** Pct */
            pct: string;
            /** Sleeve */
            sleeve: string;
        };
        /** StressResponse */
        StressResponse: {
            /**
             * As Of
             * Format: date
             */
            as_of: string;
            /** Stress */
            stress: {
                [key: string]: unknown;
            };
            /** Week */
            week: string;
        };
        /**
         * TaxRegime
         * @enum {string}
         */
        TaxRegime: "old" | "new";
        /** TermInsuranceAsset */
        TermInsuranceAsset: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            asset_type: "term_insurance";
            /**
             * Cover
             * Format: decimal
             */
            cover: string;
            /** Ends Year */
            ends_year: number;
            /** Insurer */
            insurer: string;
            /**
             * Joint
             * @default false
             */
            joint: boolean;
            /** Label */
            label?: string | null;
            /** Member Id */
            member_id?: string | null;
            /** Notes */
            notes?: string | null;
            /** Premium */
            premium?: string | null;
            /** Premium Frequency */
            premium_frequency?: ("monthly" | "quarterly" | "half_yearly" | "yearly") | null;
        };
        /** TraditionalPolicyAsset */
        TraditionalPolicyAsset: {
            /**
             * Annual Premium
             * Format: decimal
             */
            annual_premium: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            asset_type: "traditional_policy";
            /** Bonus Accrued */
            bonus_accrued?: string | null;
            /** Insurer */
            insurer: string;
            /**
             * Joint
             * @default false
             */
            joint: boolean;
            /** Label */
            label?: string | null;
            /** Maturity Illustration */
            maturity_illustration?: string | null;
            /** Member Id */
            member_id?: string | null;
            /** Notes */
            notes?: string | null;
            /** Plan */
            plan: string;
            /**
             * Policy Kind
             * @enum {string}
             */
            policy_kind: "endowment" | "money_back" | "whole_life";
            /** Premium Paying Term */
            premium_paying_term?: number | null;
            /** Start Year */
            start_year: number;
            /**
             * Sum Assured
             * Format: decimal
             */
            sum_assured: string;
            /** Surrender Value */
            surrender_value?: string | null;
            /** Term Years */
            term_years: number;
        };
        /** ULIPAsset */
        ULIPAsset: {
            /**
             * Annual Premium
             * Format: decimal
             */
            annual_premium: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            asset_type: "ulip";
            /** Equity Pct */
            equity_pct?: string | null;
            /**
             * Fund Value
             * Format: decimal
             */
            fund_value: string;
            /** Insurer */
            insurer: string;
            /**
             * Joint
             * @default false
             */
            joint: boolean;
            /** Label */
            label?: string | null;
            /** Member Id */
            member_id?: string | null;
            /** Notes */
            notes?: string | null;
            /** Start Year */
            start_year?: number | null;
        };
        /** UnvaluedAssetResponse */
        UnvaluedAssetResponse: {
            /** Asset Type */
            asset_type: string;
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /** Label */
            label: string;
            /**
             * Reason
             * @enum {string}
             */
            reason: "price_needed" | "rate_needed";
        };
        /** ValidationError */
        ValidationError: {
            /** Context */
            ctx?: Record<string, never>;
            /** Input */
            input?: unknown;
            /** Location */
            loc: (string | number)[];
            /** Message */
            msg: string;
            /** Error Type */
            type: string;
        };
        /** WeeklyPlanResponse */
        WeeklyPlanResponse: {
            /** Abroad Buy List */
            abroad_buy_list?: {
                [key: string]: unknown;
            }[];
            /** Actions */
            actions: {
                [key: string]: unknown;
            }[];
            /** Allocation */
            allocation: {
                [key: string]: {
                    [key: string]: string;
                };
            };
            /**
             * As Of
             * Format: date
             */
            as_of: string;
            /** Band Breaches */
            band_breaches: {
                [key: string]: string;
            }[];
            /**
             * Carry Forward
             * @default 0.00
             */
            carry_forward: string;
            /** Debt Answer */
            debt_answer: {
                [key: string]: unknown;
            };
            /** Debt Picks */
            debt_picks?: {
                [key: string]: unknown;
            }[];
            /**
             * Deployment Mode
             * @default personal
             * @enum {string}
             */
            deployment_mode: "personal" | "public";
            /**
             * Disclaimer
             * @default
             */
            disclaimer: string;
            /** Exit Cards */
            exit_cards: {
                [key: string]: unknown;
            }[];
            /** Foundation */
            foundation: {
                [key: string]: unknown;
            }[];
            /** Fresh */
            fresh: boolean;
            /** Highlights By Screen */
            highlights_by_screen?: {
                [key: string]: components["schemas"]["Highlight"][];
            };
            /** Inputs Digest */
            inputs_digest: string;
            /** Insurance Reviews */
            insurance_reviews?: {
                [key: string]: unknown;
            }[];
            /**
             * Me Member Id
             * Format: uuid
             */
            me_member_id: string;
            /** Members */
            members?: components["schemas"]["PlanMember"][];
            /** Notes */
            notes: string[];
            /** Placement Notes */
            placement_notes?: string[];
            /**
             * Plan Health
             * @default 100
             */
            plan_health: number;
            residency?: components["schemas"]["Residency"];
            /** Residency Notes */
            residency_notes?: string[];
            /** Route */
            route: {
                [key: string]: unknown;
            };
            /** Stress */
            stress: {
                [key: string]: unknown;
            };
            /** Unranked Options */
            unranked_options?: {
                [key: string]: unknown;
            }[];
            /** Week */
            week: string;
        };
    };
    responses: never;
    parameters: never;
    requestBodies: never;
    headers: never;
    pathItems: never;
}
export type $defs = Record<string, never>;
export interface operations {
    healthz_healthz_get: {
        parameters: {
            query?: {
                deep?: boolean;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HealthResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_accounts_v1_accounts_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AccountResponse"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    create_account_v1_accounts_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["AccountWrite"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AccountResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_account_v1_accounts__account_id__delete: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                account_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    patch_account_v1_accounts__account_id__patch: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                account_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["AccountPatch"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AccountResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_actions_v1_actions_get: {
        parameters: {
            query?: {
                week?: string | null;
            };
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ActionResponse"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    patch_action_v1_actions__action_id__patch: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                action_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ActionStatusUpdate"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ActionResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_assets_v1_assets_get: {
        parameters: {
            query?: {
                asset_type?: string | null;
                member_id?: string | null;
                status?: "active" | "closed";
            };
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AssetResponse"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    create_asset_v1_assets_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["SavingsAsset"] | components["schemas"]["FixedDepositAsset"] | components["schemas"]["RecurringDepositAsset"] | components["schemas"]["PPFAsset"] | components["schemas"]["EPFAsset"] | components["schemas"]["NPSAsset"] | components["schemas"]["PostOfficeAsset"] | components["schemas"]["BondAsset"] | components["schemas"]["GoldAsset"] | components["schemas"]["TermInsuranceAsset"] | components["schemas"]["HealthInsuranceAsset"] | components["schemas"]["TraditionalPolicyAsset"] | components["schemas"]["ULIPAsset"] | components["schemas"]["PropertyAsset"] | components["schemas"]["LoanAsset"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AssetResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    quick_add_v1_assets_quick_add_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": {
                    [key: string]: string;
                };
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["QuickAddResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_asset_v1_assets__asset_id__get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                asset_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AssetDetailResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_asset_v1_assets__asset_id__delete: {
        parameters: {
            query?: {
                hard?: boolean;
            };
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                asset_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    patch_asset_v1_assets__asset_id__patch: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                asset_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": {
                    [key: string]: unknown;
                };
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AssetResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    exchange_google_code_v1_auth_google_exchange_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": {
                    /** Code */
                    code: string;
                    /** Code Verifier */
                    code_verifier: string;
                    /** Redirect Uri */
                    redirect_uri: string;
                };
                "application/x-www-form-urlencoded": {
                    /** Code */
                    code: string;
                    /** Code Verifier */
                    code_verifier: string;
                    /** Redirect Uri */
                    redirect_uri: string;
                };
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["GoogleExchangeResponse"];
                };
            };
        };
    };
    get_brief_v1_brief__week__get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                week: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Brief"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_unsubscribe_v1_email_unsubscribe_get: {
        parameters: {
            query?: {
                token?: string | null;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "text/html": string;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    post_unsubscribe_v1_email_unsubscribe_post: {
        parameters: {
            query?: {
                token?: string | null;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "text/html": string;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_fund_check_v1_fund_check_get: {
        parameters: {
            query?: {
                member?: string;
            };
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["FundCheckResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    put_sip_override_v1_fund_check_sips__security_id__put: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                security_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["FundSipOverride"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: string;
                    };
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_goals_v1_goals_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["GoalResponse"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    create_goal_v1_goals_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Goal"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["GoalResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_goal_v1_goals__goal_id__get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                goal_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["GoalResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_goal_v1_goals__goal_id__delete: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                goal_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    patch_goal_v1_goals__goal_id__patch: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                goal_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": {
                    [key: string]: unknown;
                };
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["GoalResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_highlight_dismissals_v1_highlights_dismissals_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HighlightDismissalResponse"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    dismiss_highlight_v1_highlights__key__dismiss_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                key: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_households_v1_households_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HouseholdSummary"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_access_v1_households_access_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HouseholdAccessResponse"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    revoke_access_v1_households_access__grantee_user_id__delete: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                grantee_user_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    update_access_v1_households_access__grantee_user_id__patch: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                grantee_user_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["HouseholdAccessPatch"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HouseholdAccessResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_invites_v1_households_invites_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["InviteResponse"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    create_invite_v1_households_invites_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["InviteCreate"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["InviteCreatedResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    revoke_invite_v1_households_invites__invite_id__delete: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                invite_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    join_household_v1_households_join_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["InviteJoin"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HouseholdJoinResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    leave_household_v1_households_leave_delete: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_imports_v1_imports_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ImportJobResponse"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    upload_import_v1_imports_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "multipart/form-data": {
                    /** Format: uuid */
                    account_id?: string;
                    broker?: string;
                    doc_type?: string;
                    /** Format: binary */
                    file: string;
                    /** Format: uuid */
                    member_id?: string;
                    password?: string;
                };
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ImportUploadResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_import_v1_imports__job_id__get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                job_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ImportDetailResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    discard_import_v1_imports__job_id__delete: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                job_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    confirm_import_v1_imports__job_id__confirm_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                job_id: string;
            };
            cookie?: never;
        };
        requestBody?: {
            content: {
                "application/json": components["schemas"]["ImportConfirmationRequest"] | null;
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ImportConfirmationResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    patch_import_item_v1_imports__job_id__items__item_id__patch: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                job_id: string;
                item_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ImportItemResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    map_import_columns_v1_imports__job_id__mapping_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                job_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ImportMappingRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ImportDetailResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_inbox_v1_inbox_get: {
        parameters: {
            query?: {
                limit?: number;
            };
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["InboxResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_me_v1_me_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MeResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_me_v1_me_delete: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_checklist_v1_me_checklist_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ChecklistWrite"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    put_checklist_v1_me_checklist_put: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ChecklistWrite"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ChecklistWrite"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_consents_v1_me_consents_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ConsentValues"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    put_consents_v1_me_consents_put: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ConsentValues"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ConsentValues"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_investing_v1_me_investing_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["InvestingPrefs-Output"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    put_investing_v1_me_investing_put: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["InvestingPrefs-Input"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["InvestingPrefs-Output"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    put_profile_v1_me_profile_put: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ProfileUpsert"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProfileResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_residency_v1_me_residency_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Residency"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    put_residency_v1_me_residency_put: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ResidencyInput"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Residency"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    put_risk_answers_v1_me_risk_answers_put: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["RiskAnswersRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["RiskResultResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_members_v1_members_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MemberResponse"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    create_member_v1_members_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["MemberWrite"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MemberResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_member_v1_members__member_id__delete: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                member_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    patch_member_v1_members__member_id__patch: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                member_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["MemberPatch"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MemberResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_risk_questions_v1_meta_risk_questions_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["RiskQuestionResponse"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_networth_v1_networth_get: {
        parameters: {
            query?: {
                member_id?: string | null;
            };
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["NetworthResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_plan_v1_plan_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["WeeklyPlanResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    post_push_subscription_v1_push_subscriptions_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["PushSubscriptionInput"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["PushSubscriptionResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_push_subscription_v1_push_subscriptions__subscription_id__delete: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path: {
                subscription_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_research_ideas_v1_research_ideas_get: {
        parameters: {
            query?: {
                week?: string | null;
            };
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ResearchIdeasResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_stress_v1_stress_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Household"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["StressResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
}
